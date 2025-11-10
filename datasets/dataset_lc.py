import torch
import lmdb
import pickle
import numpy as np
import os
import torch 
from tqdm import tqdm
from torch.utils.data import Sampler


class LMDBLatentsDatasetRDMA(torch.utils.data.Dataset):
    """
    Safe for multi-process DataLoader in RDMA environments:
    - Do not pickle lmdb.Environment across processes
    - Re-open env per worker (lazy)
    - Ensure returned arrays are picklable (owning memory)
    """
    def __init__(self, lmdb_path, flip_prob=0.5, class_consist=False, min_cluster_size=32):
        self.lmdb_path = lmdb_path
        self.flip_prob = float(flip_prob)
        self.class_consistency = bool(class_consist)
        self.min_cluster_size = int(min_cluster_size)

        # DO NOT keep an open env that will be pickled to workers.
        self.env = None

        # Read metadata (num_samples) via a short-lived read-only txn
        # Use a temporary env; then close immediately.
        tmp_env = lmdb.open(
            lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
        )
        with tmp_env.begin() as txn:
            self.data_length = int(txn.get(b'num_samples').decode())
        tmp_env.close()

        # label clusters (indexes per class)
        self._label_cluster = None
        self._label_length = None
        if self.class_consistency:
            self._load_or_build_label_cluster()
            self.label_cluster_active = {k: v.copy() for k, v in self._label_cluster.items()}

    # ---------- Worker-safe LMDB open ----------
    def _ensure_env(self):
        """ Lazily (re)open LMDB env inside each worker process. """
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
            )

    # ---------- Pickle control: don't pickle LMDB env ----------
    def __getstate__(self):
        state = self.__dict__.copy()
        # Do not pickle env
        state['env'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Worker process starts with no env; it will be opened lazily
        self.env = None

    # ---------- Label cluster ----------
    def _load_or_build_label_cluster(self):
        path = os.path.join(self.lmdb_path, 'label_cluster.pt')
        if os.path.exists(path):
            loaded = torch.load(path, map_location='cpu')
            self._label_cluster = {int(k): list(map(int, v)) for k, v in loaded.items()}
            self._label_length = len(self._label_cluster)
            return

        tmp_env = lmdb.open(
            self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
        )
        label_cluster = {}
        for index in tqdm(range(self.data_length), desc='Preparing label cluster'):
            with tmp_env.begin() as txn:
                raw = txn.get(f'{index}'.encode())
                if raw is None:
                    raise IndexError(f'Index {index} is out of bounds')
                data = pickle.loads(raw)
                label = int(data['label'])
                label_cluster.setdefault(label, []).append(index)
        tmp_env.close()

        self._label_cluster = {k: list(map(int, v)) for k, v in label_cluster.items()}
        self._label_length = len(self._label_cluster)
        torch.save(self._label_cluster, path)

    def reset_label_active(self, label):
        if not self._label_cluster:
            return
        self.label_cluster_active[label] = self._label_cluster[label].copy()

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        self._ensure_env()

        with self.env.begin() as txn:
            raw = txn.get(f'{index}'.encode())
            if raw is None:
                raise IndexError(f'Index {index} is out of bounds')

            data = pickle.loads(raw)
            moments = data['moments']
            moments_flip = data['moments_flip']
            label = int(data['label'])

        use_flip = bool(torch.rand(1).item() < self.flip_prob)
        arr = moments_flip if use_flip else moments

        if isinstance(arr, np.ndarray):
            if not arr.flags['OWNDATA']:
                arr = np.array(arr, copy=True)
            arr = np.ascontiguousarray(arr)
            moments_tensor = torch.from_numpy(arr).float()
        elif torch.is_tensor(arr):
            moments_tensor = arr.detach().clone().float()  
        else:
            arr = np.array(arr, copy=True)
            arr = np.ascontiguousarray(arr)
            moments_tensor = torch.from_numpy(arr).float()

        return moments_tensor, label

    def __del__(self):
        try:
            if self.env is not None:
                self.env.close()
        except Exception:
            pass


class SameLabelBatchSampler(Sampler):
    def __init__(self, label_cluster, batch_size, min_cluster_size=32):
        self.label_cluster = label_cluster
        self.batch_size = batch_size
        self.labels = list(label_cluster.keys())
        self.min_cluster_size = min_cluster_size
        self.label_cluster_active = {k: v.copy() for k, v in self.label_cluster.items()}

    def __iter__(self):
        while True:
            label = np.random.choice(self.labels)
            if len(self.label_cluster_active[label]) < self.min_cluster_size:
                self.label_cluster_active[label] = self.label_cluster[label].copy()
            idxs = []
            for _ in range(self.batch_size):
                idxs.append(self.label_cluster_active[label].pop())
            yield idxs

    def __len__(self):
        total_samples = sum(len(v) for v in self.label_cluster.values())
        return total_samples // self.batch_size

