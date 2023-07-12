from config.config import *
import pytorch_lightning as pl
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from data.prepare import create_splits_new
if device_comp:
    device= torch.device(device_comp)

class openwebtext(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = np.memmap(self.path, dtype = np.uint16, mode = "r")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ix = torch.randint(self.__len__() - block_size, ())
        x = torch.from_numpy(self.data[ix:ix+block_size].astype(np.int64))
        y = torch.from_numpy(self.data[ix+1:ix+1+block_size].astype(np.int64))
        if device:
            x, y = x.to(device), y.to(device)
        return x, y


class data_lightning_wrapper(pl.LightningDataModule):
    def __init__(self, path, batch_size):
        super().__init__()
        self.path = path
        self.encoder = tiktoken.get_encoding('gpt2')
        self.batch_size = batch_size
        
    def process_encode(self, example):
        encodings = self.encoder.encode_ordinary(example['text'])
        encodings.append(self.encoder.eot_token)
        out = {"encodings": encodings, "length": len(encodings)}
        return out
    
    def setup(self, stage: str, make_files = False):
        
        self.train_path = os.path.join(os.getcwd(), "data/train.bin")
        self.val_path = os.path.join(os.getcwd(), "data/val.bin")
        if make_files:
            print("\n\nThis might take some time as train and val file are not created\n\n")
            num_proc = 1
            dataset = load_dataset(self.path)
            dataset = dataset['train'].train_test_split(test_size = 0.001, seed=2357, shuffle=True)
            dataset['val'] = dataset.pop('test')
            tokenized_item = dataset.map(function = self.process_encode, remove_columns = 'text', desc = "tokenizing the split", num_proc = num_proc)

            for split, dset in tokenized_item.items():
                arr_len = np.sum(dset['length'])
                filename = os.path.join(os.getcwd(), "data/{}.bin".format(split))
                arr = np.memmap(filename, dtype = np.uint16, mode = 'w+', shape = (arr_len,))

                total_batch, idx = 1024, 0
                for batch_idx in tqdm(range(total_batch), desc = "writing {}".format(filename)):
                    batch = dset.shard(num_shards=total_batch, index=batch_idx, contiguous=True).with_format('numpy')
                    arr_batch = np.concatenate(batch['encodings'])
                    arr[idx: idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                arr.flush()          
                
        if stage == 'fit':            
            self.train_data = openwebtext(self.train_path)
            self.test_data = openwebtext(self.val_path)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.batch_size, num_workers=num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size = self.batch_size, num_workers=num_workers)
