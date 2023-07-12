import tiktoken
import numpy as np
import os
import torch
from config.config import *
if device_comp:
    device= torch.device(device_comp)


def create_data_encodings(path="./data/text.txt", encoder_config = "gpt2"):
    with open(path, "rb") as fo:
        data = fo.read().decode('utf-8')

    data_size = len(data)
    train_data_size, test_data_size = int(data_size * 0.9), int(data_size * 0.1)
    train_data = data[:train_data_size]
    test_data = data[train_data_size:]

    encoder = tiktoken.encoding_for_model(encoder_config)
    encoded_data_train = encoder.encode(train_data)
    encoded_data_test = encoder.encode(test_data)

    encoded_data_train = np.array(encoded_data_train, np.int64)
    encoded_data_test = np.array(encoded_data_test, np.int64)
    encoded_data_train.tofile(os.path.join(os.getcwd(), "data/train.bin"))
    encoded_data_test.tofile(os.path.join(os.getcwd(), "data/test.bin"))

    return encoded_data_train, encoded_data_test, encoder


class charEncoder():
    def __init__(self, path):
        with open(path, "rb") as fo:
            self.data = fo.read().decode('utf-8')
        self.char = sorted(list(set(self.data)))
        self.stoi = {ch:i for i, ch in enumerate(self.char)}
        self.itos = {i:ch for i, ch in enumerate(self.char)}

    def encode(self, char_list: list):
        return [self.stoi[i] for i in char_list]
    
    def decode(self, int_list: list):
        return ''.join([self.itos[i] for i in int_list])
    
    def n_vocab(self):
        return len(self.char)
    
def get_charEncoding(path):
    encoder = charEncoder(path)
    with open(path, "rb") as fo:
        data = fo.read().decode('utf-8')
    data_size = len(data)
    train_data_size, test_data_size = int(data_size * 0.9), int(data_size * 0.1)
    train_data = data[:train_data_size]
    test_data = data[train_data_size:]

    encoded_data_train = np.array(encoder.encode(train_data), dtype=np.int64)
    encoded_data_test = np.array(encoder.encode(test_data), dtype=np.int64)
    return encoded_data_train, encoded_data_test, encoder


def create_splits(train_df, test_df, mode):
    data = train_df if mode == 'train' else test_df
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y

def create_splits_new(df_to_split):
    ix = torch.randint(len(df_to_split) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(df_to_split[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(df_to_split[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device:
        x, y = x.to(device), y.to(device)
    return x, y



