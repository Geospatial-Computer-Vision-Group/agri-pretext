# %%
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from pathlib import Path
import zarr

# %%
DATA_DIR = "/home/moti/sickle_s2/sickle_s2.zarr"
# %%
class SickleS2Pretrain(Dataset):
    def __init__(self, root):
        self.ds = zarr.open(root)
        self.idxs = list(self.ds.group_keys())
        
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, i):
        return torch.from_numpy(np.transpose(self.ds[self.idxs[i]]['data'][:][:,[2,1,0,3]], (1, 0, 2 , 3)).reshape(4,-1)) # R G B NIR

# %%
ds = SickleS2Pretrain(DATA_DIR)

# %%
class RunningStatsButFast(torch.nn.Module):
    def __init__(self, shape, dims):
        """Initializes the RunningStatsButFast method.
        
        A PyTorch module that can be put on the GPU and calculate the multidimensional
        mean and variance of inputs online in a numerically stable way. This is useful
        for calculating the channel-wise mean and variance of a big dataset because you
        don't have to load the entire dataset into memory.
        
        Uses the "Parallel algorithm" from: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        Similar implementation here: https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py#L5
        
        Access the mean, variance, and standard deviation of the inputs with the
        `mean`, `var`, and `std` attributes.
        
        Example:
        ```
        rs = RunningStatsButFast((12,), [0, 2, 3])
        for inputs, _ in dataloader:
            rs(inputs)
        print(rs.mean)
        print(rs.var)
        print(rs.std)
        ```
        
        Args:
            shape: The shape of resulting mean and variance. For example, if you
                are calculating the mean and variance over the 0th, 2nd, and 3rd
                dimensions of inputs of size (64, 12, 256, 256), this should be 12.
            dims: The dimensions of your input to calculate the mean and variance
                over. In the above example, this should be [0, 2, 3].
        """
        super(RunningStatsButFast, self).__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.register_buffer('std', torch.ones(shape))
        self.register_buffer('count', torch.zeros(1))
        self.register_buffer('min', torch.inf * torch.ones(shape))
        self.register_buffer('max', torch.zeros(shape))
        self.dims = dims

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, dim=self.dims)
            batch_var = torch.var(x, dim=self.dims)
            batch_count = torch.tensor(x.shape[self.dims[0]], dtype=torch.float)
            batch_min = torch.min(x,dim=self.dims[0]).values
            batch_max = torch.max(x,dim=self.dims[0]).values

            n_ab = self.count + batch_count
            m_a = self.mean * self.count
            m_b = batch_mean * batch_count
            M2_a = self.var * self.count
            M2_b = batch_var * batch_count

            delta = batch_mean - self.mean

            self.mean = (m_a + m_b) / (n_ab)
            # we don't subtract -1 from the denominator to match the standard Numpy/PyTorch variances
            self.var = (M2_a + M2_b + delta ** 2 * self.count * batch_count / (n_ab)) / (n_ab)
            self.count += batch_count
            self.std = torch.sqrt(self.var + 1e-8)
            self.min = torch.minimum(self.min,batch_min)
            self.max = torch.maximum(self.max,batch_max)

    def forward(self, x):
        self.update(x)
        return x

# %%
stats = RunningStatsButFast((4,),[1]).cuda()

# %%
for i,x in enumerate(tqdm(ds)):
    x = x.cuda()
    stats(x)

# %%
torch.save(stats.state_dict(),"stats.pth")


