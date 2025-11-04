import torch
import lmdb
import pickle
import numpy as np
import os
import torch 
from tqdm import tqdm
from torch.utils.data import Sampler

class LMDBLatentsDataset(torch.utils.data.Dataset):
    """    
    Args:
        lmdb_path (str): LMDB dataset path.
        flip_prob (float): flip or upflip.
    """
    def __init__(self, lmdb_path, flip_prob=0.5, class_consist=False, min_cluster_size=32):
        self.env = lmdb.open(lmdb_path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
        self.lmdb_path = lmdb_path
        
        with self.env.begin() as txn:
            self.data_length = int(txn.get('num_samples'.encode()).decode())
        self.flip_prob = flip_prob
        self.class_consist = class_consist
        self.min_cluster_size = min_cluster_size
        if class_consist:
            self.get_label_cluster()
            self.label_cluster_active = {k: v.copy() for k, v in self.label_cluster.items()}

    def reset_label_active(self, label):
        self.label_cluster_active[label] = self.label_cluster[label].copy()


    def get_label_cluster(self):
        label_cluster_path = os.path.join(self.lmdb_path, 'label_cluster.pt')
        if os.path.exists(label_cluster_path):
            self.label_cluster = torch.load(label_cluster_path)
            self.label_length = len(self.label_cluster)
            return
        
        label_cluster = {}
        for index in tqdm(range(self.data_length), desc='Preparing label cluster'):
            with self.env.begin() as txn:
                data = txn.get(f'{index}'.encode())
                if data is None:
                    raise IndexError(f'Index {index} is out of bounds')
                
                data = pickle.loads(data)
                label = data['label']
                if label not in label_cluster:
                    label_cluster[label] = []
                label_cluster[label].append(index)
        self.label_cluster = label_cluster
        self.label_length = len(label_cluster)
        torch.save(self.label_cluster, os.path.join(self.lmdb_path, 'label_cluster.pt'))

    def __len__(self):
        return self.data_length 
    
    def __getitem__(self, index):
        with self.env.begin() as txn:
            data = txn.get(f'{index}'.encode())
            if data is None:
                raise IndexError(f'Index {index} is out of bounds')
            
            data = pickle.loads(data)
            moments = data['moments']
            moments_flip = data['moments_flip']
            label = data['label']
            
            use_flip = torch.rand(1).item() < self.flip_prob
            
            moments_to_use = moments_flip if use_flip else moments
            
            moments_tensor = torch.from_numpy(moments_to_use).float()
            
            return moments_tensor, label
    
    def __del__(self):
        self.env.close()


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

