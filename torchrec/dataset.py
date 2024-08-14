
import os
import hashlib
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.utils import murmurhash3_32
import torch
import torch.utils
from torch.utils.data import Dataset, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from joblib import Parallel, delayed

pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_rows = 999
pd.options.display.max_columns = 100

def pad_list(arr_list, padding_value, max_len=None):
    '''
    arr_list: list/array of np.array
    '''
    if max_len is None:
        max_len = max([len(arr) for arr in arr_list])

    for k, arr in enumerate(arr_list):
        if len(arr) < max_len:
            arr_list[k] = np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=padding_value)
        else:
            arr_list[k] = arr[:max_len]
    return arr_list

class FeatureTransformer:
    def __init__(self, 
                 feat_configs, 
                 category_force_hash=False, 
                 category_dynamic_vocab=True,
                 category_min_freq=None,
                 category_upper_lower_sensitive=True,
                 numerical_update_stats=False,
                 list_padding_value=None,
                 list_padding_maxlen=None,
                 outliers_category=['','None','none','nan','NaN','NAN','NaT','unknown','Unknown','Other','other','others','Others','REJ','Reject','REJECT','Rejected'], 
                 outliers_numerical=[], 
                 verbose=False):
        """
        Feature transforming for both train and test dataset.
        Args:
            feat_configs: list of dict, feature configurations. for example, 
                [
                    {'name': 'a', 'dtype': 'numerical', 'norm': 'std'},   # 'norm' in ['std','[0,1]']
                    {'name': 'a', 'dtype': 'numerical', 'hash_buckets': 10, emb_dim: 8}, # Discretization
                    {'name': 'b', 'dtype': 'category', 'emb_dim': 8, 'hash_buckets': 100}, # category feature with hash_buckets
                    {'name': 'c', 'dtype': 'category', 'islist': True, 'emb_dim': 8, 'maxlen': 256}, # sequence feature, if maxlen is set, it will truncate the sequence to the maximum length
                ]
            is_train: bool, whether it's training dataset
            category_force_hash: bool, whether to force hash all category features, which will be useful for large category features and online learning scenario, only effective when is_train=True
            category_dynamic_vocab: bool, whether to use dynamic vocab for category features, only effective when is_train=True
            category_min_freq: int, minimum frequency for category features, only effective when is_train=True
            numerical_update_stats: bool, whether to update mean, std, min, max for numerical features, only effective when is_train=True
            outliers_category: list, outliers for category features
            outliers_numerical: list, outliers for numerical features
            verbose: bool, whether to print the processing details
            n_jobs: int, number of parallel jobs
        """
        self.feat_configs = feat_configs
        self.category_force_hash = category_force_hash
        self.category_dynamic_vocab = category_dynamic_vocab
        self.category_min_freq = category_min_freq
        self.numerical_update_stats = numerical_update_stats
        self.outliers_category = outliers_category
        self.outliers_numerical = outliers_numerical
        self.category_upper_lower_sensitive = category_upper_lower_sensitive
        self.list_padding_value = list_padding_value
        self.list_padding_maxlen = list_padding_maxlen
        self.verbose = verbose

        assert all([f['dtype'] in ['category', 'numerical'] for f in self.feat_configs]), 'Only support category and numerical features'
        assert not (self.category_dynamic_vocab and self.category_force_hash), 'category_dynamic_vocab and category_force_hash cannot be set at the same time'

    def transform(self, df, is_train=False, n_jobs=1):
        """
        Transforms the DataFrame based on the feature configurations.
        Args:
            df: pandas DataFrame
        Returns:
            df: pandas DataFrame, transformed dataset
        """
        if self.verbose:
            print(f'==> Feature transforming (is_train={is_train}), note that feat_configs will be updated when is_train=True...')

        if n_jobs <= 1:
            for k, f in enumerate(self.feat_configs):
                df[f['name']], updated_f = self._transform_one(df[f['name']], f, is_train)
                if is_train:
                    self.feat_configs[k] = updated_f
            return df

        # parallel process features
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._transform_one)(df[f_config['name']], f_config, is_train) for f_config in self.feat_configs
        )

        # update df & feat_configs
        for k, (updated_s, updated_f) in zip(range(len(self.feat_configs)), results):
            df[updated_f['name']] = updated_s
            if is_train:
                self.feat_configs[k] = updated_f

        return df

    def hash(self, v, hash_buckets):
        """
        Hash function for category features.
        """
        # hash_object = hashlib.sha256(str(v).encode())
        # hash_digest = hash_object.hexdigest()
        # hash_integer = int(hash_digest, 16)

        hash_integer = murmurhash3_32(str(v), seed=42, positive=True)
        return hash_integer % hash_buckets

    def update_meanstd(self, s, his_freq_cnt=0, mean=None, std=None):
        """
        Update mean, std for numerical feature.
        If none, calculate from s, else update the value by new input data.
        """
        s_mean = s.mean()
        s_std = s.std()

        # update mean and std
        mean = s_mean if mean is None else (mean * his_freq_cnt + s_mean * len(s)) / (his_freq_cnt + len(s))
        std = s_std if std is None else np.sqrt((his_freq_cnt * (std ** 2) + len(s) * (s_std ** 2) + his_freq_cnt * (mean - s_mean) ** 2) / (his_freq_cnt + len(s)))

        return mean, std

    def update_minmax(self, s, min_val=None, max_val=None):
        """
        Update min, max for numerical feature.
        If none, calculate from s, else update the value by new input data.
        """
        s_min = s.min()
        s_max = s.max()

        # update min and max
        min_val = s_min if min_val is None else min(min_val, s_min)
        max_val = s_max if max_val is None else max(max_val, s_max)

        return min_val, max_val

    def process_category(self, feat_config, s, is_train=False):
        """
        Process category features.
        """
        name = feat_config['name']
        oov = feat_config.get('oov', 'other')  # out of vocabulary

        outliers_category = feat_config.get('outliers', self.outliers_category)
        s = s.replace(outliers_category, np.nan).fillna(oov).map(lambda x: str(int(x) if type(x) is float else x))
        s = s.astype(str)

        category_upper_lower_sensitive = feat_config.get('upper_lower_sensitive', self.category_upper_lower_sensitive)
        if not category_upper_lower_sensitive:
            s = s.str.lower()

        hash_buckets = feat_config.get('hash_buckets')
        if self.category_force_hash and hash_buckets is None:
            hash_buckets = s.nunique()
            # Auto choose the experienced hash_buckets for embedding table to avoid hash collision
            if hash_buckets < 100:
                hash_buckets *= 10
            elif hash_buckets < 1000:
                hash_buckets = max(hash_buckets * 5, 1000)
            elif hash_buckets < 10000:
                hash_buckets = max(hash_buckets * 2, 5000)
            elif hash_buckets < 1000000:
                hash_buckets = max(hash_buckets, 20000)
            else:
                hash_buckets = hash_buckets // 10

            if self.verbose:
                print(f'Forcing hash category {name} with hash_buckets={hash_buckets}...')

            if is_train:
                feat_config['hash_buckets'] = hash_buckets

        category_dynamic_vocab = feat_config.get('dynamic_vocab', self.category_dynamic_vocab)
        assert not (category_dynamic_vocab and hash_buckets), 'dynamic_vocab and hash_buckets cannot be set at the same time for feature: {name}'

        if is_train:
            # update feat_config
            feat_config['type'] = 'sparse'

            # low frequency category filtering
            raw_vocab = s.value_counts()
            min_freq = feat_config.get('min_freq', self.category_min_freq)
            if min_freq:
                raw_vocab = raw_vocab[raw_vocab >= min_freq]
            
            # check if it's dask series
            # if isinstance(raw_vocab, dd.Series):
            #     raw_vocab = raw_vocab.compute()

        if hash_buckets:
            if self.verbose:
                print(f'Hashing category {name} with hash_buckets={hash_buckets}...')
            if is_train:
                # update feat_config
                feat_config['num_embeddings'] = hash_buckets
                if min_freq:
                    feat_config['vocab'] = {v: freq_cnt for k, (v, freq_cnt) in enumerate(raw_vocab.items())}

            if 'vocab' in feat_config:
                s = s.map(lambda x: x if x in feat_config['vocab'] else oov)
            s = s.map(lambda x: self.hash(x, hash_buckets)).astype(int)
        else:
            if self.verbose:
                print(f'Converting category {name} to indices...')
            if is_train:
                if 'vocab' not in feat_config:
                    feat_config['vocab'] = {}
                    idx = 0

                    category_dynamic_vocab = True # force to dynamic vocab when no vocab is provided
                else:
                    idx = max([v['idx'] for v in feat_config['vocab'].values()])

                # update dynamic vocab (should combine with dynamic embedding module when online training)
                for k, (v, freq_cnt) in enumerate(raw_vocab.items()):
                    if v not in feat_config['vocab'] and category_dynamic_vocab:
                        idx += 1
                        feat_config['vocab'][v] = {'idx': idx, 'freq_cnt': freq_cnt}
                    elif v in feat_config['vocab']:
                        feat_config['vocab'][v]['freq_cnt'] += freq_cnt

                if oov not in feat_config['vocab']:
                    feat_config['vocab'][oov] = {'idx': 0, 'freq_cnt': 0}

                if self.verbose:
                    print(f'Feature {name} vocab size: {feat_config.get("num_embeddings")} -> {len(feat_config["vocab"])}')

                feat_config['num_embeddings'] = idx + 1

            # convert to indices
            oov_index = feat_config['vocab'].get(oov)
            s = s.map(lambda x: feat_config['vocab'].get(x, oov_index)['idx']).astype(int)

        return s, feat_config

    def process_numerical(self, feat_config, s, is_train=False):
        """
        Process numerical features.
        """
        hash_buckets = feat_config.get('hash_buckets', None)
        emb_dim = feat_config.get('emb_dim', None)
        discretization = True if (hash_buckets or emb_dim) else False
        normalize = feat_config.get('norm', None)
        if normalize:
            assert normalize in ['std', '[0,1]'], f'Unsupported norm: {normalize}'
        assert not (discretization and normalize), f'hash_buckets/emb_dim and norm cannot be set at the same time: {feat_config}'

        if is_train:
            # update mean, std, min, max
            feat_config['type'] = 'sparse' if discretization else 'dense'

            if 'mean' not in feat_config or 'std' not in feat_config or self.numerical_update_stats:
                feat_config['mean'], feat_config['std'] = self.update_meanstd(s, feat_config.get('freq_cnt', 0), mean=feat_config.get('mean'), std=feat_config.get('std'))
                feat_config['freq_cnt'] = feat_config.get('freq_cnt', 0) + len(s)

            if 'min' not in feat_config or 'max' not in feat_config or self.numerical_update_stats:
                feat_config['min'], feat_config['max'] = self.update_minmax(s, min_val=feat_config.get('min'), max_val=feat_config.get('max'))

            if self.verbose:
                print(f'Feature {feat_config["name"]} mean: {feat_config["mean"]}, std: {feat_config["std"]}, min: {feat_config["min"]}, max: {feat_config["max"]}')

            if discretization:
                hash_buckets = 10 if hash_buckets is None else hash_buckets

                bins = np.percentile(s[s.notna()], q=np.linspace(0, 100, num=hash_buckets))
                non_adjacent_duplicates = np.append([True], np.diff(bins) != 0)
                feat_config['vocab'] = list(bins[non_adjacent_duplicates])

                feat_config['vocab'] = [np.NaN, float('-inf')] + feat_config['vocab'] + [float('inf')]

        if normalize == 'std':
            oov = feat_config.get('oov', feat_config['mean'])
            s = (s.fillna(oov) - feat_config['mean']) / feat_config['std']
        elif normalize == '[0,1]':
            oov = feat_config.get('oov', feat_config['mean'])
            s = (s.fillna(oov) - feat_config['min']) / (feat_config['max'] - feat_config['min'] + 1e-12)
        elif discretization:
            bins = [v for v in feat_config['vocab'] if not np.isnan(v)]
            s = pd.cut(s, bins=bins, labels=False, right=True) + 1
            s = s.fillna(0).astype(int)  # index 0 is for nan values

        return s, feat_config

    def process_list(self, feat_config, s, is_train=False):
        """
        Process list features.
        """
        dtype = feat_config['dtype']

        # if column is string type, split by comma, make sure no space between comma
        if isinstance(s.iat[0], str):
            if self.verbose:
                print(f'Feature {feat_config["name"]} is a list feature but input string type, split it by comma...')
            s = s.str.split(',')
            if dtype == 'numerical':
                s = s.map(lambda x: [float(v) for v in x if v])
        
        max_len = feat_config.get('maxlen', self.list_padding_maxlen)
        if max_len:
            s = s.map(lambda x: x[:max_len] if isinstance(x, list) else x)
        flat_s = s.explode()
        if dtype == 'category':
            flat_s, updated_f = self.process_category(feat_config, flat_s, is_train)
        elif dtype == 'numerical':
            flat_s, updated_f = self.process_numerical(feat_config, flat_s, is_train)
        else:
            raise ValueError(f'Unsupported data type: {dtype}')
        s = flat_s.groupby(level=0).agg(list)
        # padding
        padding_value = feat_config.get('padding_value', self.list_padding_value)
        if padding_value and dtype == 'category':
            max_len = min([s.map(len).max(), max_len]) if max_len else s.map(len).max()
            s = s.map(lambda x: x + [padding_value] * (max_len - len(x)) if len(x) < max_len else x[:max_len])
        return s, updated_f

    def _transform_one(self, s, f, is_train=False):
        """
        Transform a single feature based on the feature configuration.
        """
        fname = f['name']
        dtype = f['dtype']
        islist = f.get('islist', None)
        pre_transform = f.get('pre_transform', None)

        # pre-process
        if pre_transform:
            s = s.map(pre_transform)

        if self.verbose:
            print(f'Processing feature {fname}...')

        if islist:
            updated_s, updated_f = self.process_list(f, s, is_train)
        elif dtype == 'category':
            updated_s, updated_f = self.process_category(f, s, is_train)
        elif dtype == 'numerical':
            updated_s, updated_f = self.process_numerical(f, s, is_train)
        else:
            raise ValueError(f'Unsupported data type: {dtype}')

        return updated_s, updated_f
    
    def get_feat_configs(self):
        return self.feat_configs

class DataFrameDataset(Dataset):
    '''
    Var-length supported pytorch dataset for DataFrame.
    '''
    def __init__(self, df, feat_configs, target_cols=None, is_raw=True, **kwargs):
        """
        Args:
            df: pandas DataFrame
            feat_configs: list of dict, feature configurations. for example, 
                [
                    {'name': 'a', 'dtype': 'numerical', 'norm': 'std'},   # 'norm' in ['std','[0,1]']
                    {'name': 'a', 'dtype': 'numerical', 'hash_buckets': 10, emb_dim: 8}, # Discretization
                    {'name': 'b', 'dtype': 'category', 'emb_dim': 8, 'hash_buckets': 100}, # category feature with hash_buckets
                    {'name': 'c', 'dtype': 'category', 'islist': True, 'emb_dim': 8}, # sequence feature
                ]
            target_cols: list of str, target columns
            is_raw: bool, whether the input DataFrame is raw data without feature transforming
            kwargs: include 
                - FeatureTransformer parameters when is_raw=True
                    n_jobs: int, number of parallel jobs
                    category_force_hash: bool, whether to force hash all category features
                    category_upper_lower_sensitive: bool, whether category features are upper/lower case sensitive
                    numerical_update_stats: bool, whether to update mean, std, min, max for numerical features
                    outliers_category: list, outliers for category features
                    outliers_numerical: list, outliers for numerical features
                    verbose: bool, whether to print the processing details
                - list padding parameters
                    list_padding_value: int, padding value for list features, if not set, default to -100
                    list_padding_maxlen: int, maximum length for padding list features, if not set, default to min(256, max_len)
                    list_padding_in_collate_fn: bool, whether to padding in collate_fn, default to True

        Example of using it in DataLoader:
            ```python
            from torch.utils.data import DataLoader
            from torchrec import DataFrameDataset

            ds = DataFrameDataset(df, feat_configs, target_cols, is_raw=True, n_jobs=8)
            DataLoader(ds, collate_fn=ds.collate_fn, batch_size=32, shuffle=True)
            ```
        """
        n_jobs = kwargs.get('n_jobs', 1) # os.cpu_count()
        verbose = kwargs.get('verbose', False)

        self.list_padding_value = kwargs.get('list_padding_value', -100)
        self.list_padding_maxlen = kwargs.get('list_padding_maxlen', 256)
        self.list_padding_in_collate_fn = kwargs.get('list_padding_in_collate_fn', True) # whether to padding in collate_fn
        
        if is_raw:
            assert 'is_train' in kwargs, 'is_train parameter should be provided when is_raw=True'
            is_train = kwargs['is_train']
            
            self.transformer = FeatureTransformer(
                feat_configs,
                category_force_hash=kwargs.get('category_force_hash', False),
                category_upper_lower_sensitive=kwargs.get('category_upper_lower_sensitive', True),
                numerical_update_stats=kwargs.get('numerical_update_stats', False),
                list_padding_value=self.list_padding_value if not self.list_padding_in_collate_fn else None,
                list_padding_maxlen=self.list_padding_maxlen if not self.list_padding_in_collate_fn else None,
                outliers_category=kwargs.get('outliers_category', []),
                outliers_numerical=kwargs.get('outliers_numerical', []),
                verbose=verbose
            )
            
            df = self.transformer.transform(df, is_train=is_train, n_jobs=n_jobs)
            if verbose:
                print(f'==> Feature transforming (is_train={is_train}) done...')

        self.dense_cols = [f['name'] for f in feat_configs if f['type'] == 'dense' and not f.get('islist')]
        self.seq_dense_cols = [f['name'] for f in feat_configs if f['type'] == 'dense' and f.get('islist')]
        self.sparse_cols = [f['name'] for f in feat_configs if f['type'] == 'sparse' and not f.get('islist')]
        self.seq_sparse_cols = [f['name'] for f in feat_configs if f['type'] == 'sparse' and f.get('islist')]
        self.weight_cols_mapping = {f['name']: f.get('weight_col') for f in feat_configs if f['type'] == 'sparse' if f.get('weight')}    
        self.target_cols = target_cols

        self.seq_sparse_configs = {f['name']: f for f in feat_configs if f['type'] == 'sparse' and f.get('islist')}

        if verbose:
            print(f'==> Dense features: {self.dense_cols}')
            print(f'==> Sparse features: {self.sparse_cols}')
            print(f'==> Sequence dense features: {self.seq_dense_cols}')
            print(f'==> Sequence sparse features: {self.seq_sparse_cols}')
            print(f'==> Weight columns mapping: {self.weight_cols_mapping}')
            print(f'==> Target columns: {self.target_cols}')

        self.total_samples = len(df)

        self.convert_to_numpy(df)
        # padding for sequences
        if not is_raw and not self.list_padding_in_collate_fn: # not padded in the transforming process and will not in the collate_fn
            for col in self.seq_sparse_cols:
                list_padding_value = self.seq_sparse_configs[col].get('padding_value', self.list_padding_value)
                list_padding_maxlen = self.seq_sparse_configs[col].get('maxlen', self.list_padding_maxlen)
                list_padding_maxlen = min(
                    list_padding_maxlen,
                    max([len(v) for v in self.seq_sparse_data[col]])
                )
                self.seq_sparse_data[col] = pad_list(
                    self.seq_sparse_data[col], list_padding_value, list_padding_maxlen
                )
                if col in self.weight_cols_mapping:
                    self.weight_data[f'{col}_wgt'] = pad_list(
                        self.weight_data[f'{col}_wgt'], 0., list_padding_maxlen
                    )

        if verbose:
            print(f'==> Finished dataset initialization, total samples: {self.total_samples}')

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        features = {}
        if hasattr(self, 'dense_data'):
            data = self.dense_data[idx]
            # if isinstance(self.dense_data, dd.Series):
            #     data = data.compute()
            features.update( {'dense_features': data} )
        if hasattr(self, 'seq_dense_data'):
            data = self.seq_dense_data[idx]
            # if isinstance(self.seq_dense_data, dd.Series):
            #     data = data.compute()
            features.update( {'seq_dense_features': data} )
        
        all_sparse_data = {**self.sparse_data, **self.seq_sparse_data, **self.weight_data}
        for k,v in all_sparse_data.items():
            features[f'{k}'] = v[idx]

        if hasattr(self, 'target'):
            return features, self.target[idx]
        else:
            return features

    def convert_to_numpy(self, df):
        if self.dense_cols is not None and len(self.dense_cols) > 0:
            self.dense_data = df[self.dense_cols].to_numpy(dtype=np.float32)
        
        if self.seq_dense_cols is not None and len(self.seq_dense_cols) > 0: 
            self.seq_dense_data = df[self.seq_dense_cols].applymap(
                lambda x: np.array([np.nansum(x), np.nanmax(x), np.nanmin(x)])
            ).to_numpy(dtype=np.float32)

        self.weight_data = {}
        self.sparse_data = {}
        for col in self.sparse_cols:
            self.sparse_data[col] = df[[col]].to_numpy(dtype=np.int32)
            if col in self.weight_cols_mapping:
                weight_col = self.weight_cols_mapping[col]
                self.weight_data[f'{col}_wgt'] = df[[weight_col]].to_numpy(dtype=np.float32)  

        # for sparse sequences, padding to the maximum length
        self.seq_sparse_data = {}
        for col, cfg in self.seq_sparse_configs.items():
            # convert to np.array if it's list type
            if not isinstance(df[col].iloc[0], np.ndarray):
                df[col] = df[col].map(np.array)
            if col in self.weight_cols_mapping:
                weight_col = self.weight_cols_mapping[col]
                if not isinstance(df[weight_col].iloc[0], np.ndarray):
                    df[weight_col] = df[weight_col].map(np.array)

            # return array of arrays
            self.seq_sparse_data[col] = df[col].to_numpy()
            if col in self.weight_cols_mapping:
                weight_col = self.weight_cols_mapping[col]
                self.weight_data[f'{col}_wgt'] = df[weight_col].to_numpy()

        if self.target_cols is not None:
            self.target = df[self.target_cols].to_numpy(dtype=np.float32)

    def collate_fn(self, batch):
        """
        Collate function for DataFrameDataset.
        It mainly for padding sequences if not padded. If there aren't any sequences, it will call the default collate_fn.
        Args:
            batch: list of tuples, each tuple contains features and target
        Returns:
            batch: list of tensors, each tensor contains features and target
        """
        if not self.list_padding_in_collate_fn:
            return torch.utils.data.dataloader.default_collate(batch)

        if len(batch) == 0:
            return batch
        
        if len(batch[0]) == 2:
            features, target = zip(*batch)
        else:
            features = batch

        batch_feat_keys = set( [k for k in features[0].keys()] )

        # padding for sequences if not padded
        # TODO: how to speed up this process?
        for col in self.seq_sparse_cols:
            wgt_col = f'{col}_wgt'

            list_padding_maxlen = self.seq_sparse_configs[col].get('maxlen', self.list_padding_maxlen)
            list_padding_value = self.seq_sparse_configs[col].get('padding_value', self.list_padding_value)

            max_len = max([len(f[col]) for f in features])
            if max_len <= 1:  # sequence features that length equal to 1 in all samples
                continue
            max_len = min([list_padding_maxlen, max_len]) if list_padding_maxlen else max_len

            # iterate over each sample
            for k, f in enumerate(features):
                updated_f = np.pad(f[col], (0, max_len - len(f[col])), 'constant', constant_values=list_padding_value) if len(f[col]) < max_len else f[col][:max_len]
                features[k].update({col: torch.tensor(updated_f, dtype=torch.int)})
                if wgt_col in batch_feat_keys:
                    updated_w = np.pad(f[wgt_col], (0, max_len - len(f[wgt_col])), 'constant', constant_values=0.) if len(f[f'{col}_wgt']) < max_len else f[f'{col}_wgt'][:max_len]
                    features[k].update({wgt_col: torch.tensor(updated_w, dtype=torch.float)})

        # call the original collate_fn
        batch = list(zip(features, target)) if len(batch[0]) == 2 else features
        return torch.utils.data.dataloader.default_collate(batch)

def get_dataloader(ds: DataFrameDataset, **kwargs):
    '''
    Get DataLoader from DataFrameDataset.
    '''
    return torch.utils.data.DataLoader(ds, collate_fn=ds.collate_fn, **kwargs)
