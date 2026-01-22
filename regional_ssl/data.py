import zarr
import numpy as np
import torch
import os
from pathlib import Path
from torch.utils.data import Dataset,ConcatDataset, random_split, default_collate,DataLoader
import lightning as L
from tqdm import tqdm
from serialize import TorchSerializedList
from multiprocessing import shared_memory

DATA_DIR = "/home/moti/sickle_s2/sickle_s2.zarr"
FREQ_DIR = "/home/moti/sickle_s2/freq"

def get_ts(ds,i):
    data = ds[i]['data']
    timestamps = ds[i]['timestamps'][:]
    return data, timestamps

def get_month_year(dates):
    return dates.astype('datetime64[M]').astype(int)%12, dates.astype('datetime64[Y]').astype(int) + 1970

def create_date_idxs(dates):
    n = len(dates)
    indices = np.triu_indices(n)    
    m1,y1 = get_month_year(dates[indices[0]])
    m2,y2 = get_month_year(dates[indices[1]])
    date_diff = m2 - m1 + 12*(y2-y1)
    # Create mask for valid pairs diff <= 3
    valid_mask = (date_diff <= 3)
    # Get indices of valid pairs
    valid_indices = np.column_stack(indices)[valid_mask]
    return valid_indices

def collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = default_collate([item[key] for item in batch])
    return collated_batch

class SicklePretrainDataset(Dataset):
    def __init__(self,data_dir,indexes,use_freq=False):
        self.data_dir = data_dir
        self.indexes = indexes
        self.min_year = 2002 # hard-coded, like SatMAE
        self.use_freq = use_freq

    def __len__(self):
        return len(self.indexes)
    
    def _load_freq_data(self, _id):
        freq_path = os.path.join("/".join(self.data_dir.split("/")[:-1]), f"freq/sample_{_id}_top3freq.npy")
        return np.load(freq_path, mmap_mode='r')
    def __getitem__(self,idx):
        _id,t1,t2 = self.indexes[idx]
        ds = zarr.open(self.data_dir)
        data = ds[f"sample_{_id}"]['data']   
        chip1 = data[t1,:4,...][:].astype(np.float32)
        chip2 = data[t2,:4,...][:].astype(np.float32)
        month1,year1 = get_month_year(ds[f"sample_{_id}"]['timestamps'][t1])
        month2,year2 = get_month_year(ds[f"sample_{_id}"]['timestamps'][t2])
        sample = {
            'chip1':chip1,
            'chip2':chip2,
            'year1':year1,
            'year2':year2,
            'month1':month1,
            'month2':month2
        }
        if self.use_freq:
            sample['chip_freq'] = self._load_freq_data(_id)
        return sample

class SicklePretrainDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=8, split_ratio=0.8,use_freq=False):
        super().__init__()
        self.data_dir = data_dir
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_freq = use_freq

    def setup(self,stage):
        ids = range(6602)
        ds = zarr.open(self.data_dir,mode='r')
        indexes = []
        for _id in tqdm(ids):
            ts = ds[f"sample_{_id}"]['timestamps'][:] 
            valid_idxs = create_date_idxs(ts) 
            id_repeats = [[_id, t1, t2] for t1, t2 in valid_idxs]
            indexes.extend(id_repeats)
        self.indexes = TorchSerializedList(indexes)
        self.dataset = SicklePretrainDataset(self.data_dir, self.indexes, self.use_freq)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [self.split_ratio,1-self.split_ratio])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            collate_fn=collate_fn,
            prefetch_factor=2,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            collate_fn=collate_fn,
            prefetch_factor=2,
            persistent_workers=True
        )

class SickleMAEPretrainDataset(Dataset):
    def __init__(self,data_dir,indexes):
        self.data_dir = data_dir
        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)
    
    def __getitem__(self,idx):
        _id,t1 = self.indexes[idx]
        ds = zarr.open(self.data_dir)
        data = ds[f"sample_{_id}"]['data']   
        chip = data[t1,:4,...][:].astype(np.float32)
        sample = {
            'chip':chip,
        }
        return sample
    
class SickleMAEPretrainDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=8, split_ratio=0.8):
        super().__init__()
        self.data_dir = data_dir
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self,stage):
        ids = range(6602)
        ds = zarr.open(self.data_dir,mode='r')
        indexes = []
        for _id in tqdm(ids):
            ts = ds[f"sample_{_id}"]['timestamps'][:] 
            valid_idxs = range(len(ts))
            id_repeats = [[_id, t1] for t1 in valid_idxs]
            indexes.extend(id_repeats)
        self.indexes = TorchSerializedList(indexes)
        self.dataset = SickleMAEPretrainDataset(self.data_dir, self.indexes)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [self.split_ratio,1-self.split_ratio])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            collate_fn=collate_fn,
            prefetch_factor=2,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            collate_fn=collate_fn,
            prefetch_factor=2,
            persistent_workers=True
        )