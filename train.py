import numpy as np
import pandas as pd
from data.prepare import get_charEncoding, create_splits, create_data_encodings
from config.config import *
from model.model import GPTModel
from model.utilities import evaluate_loss
import torch
import torch.nn as nn
from torch.nn import functional as F
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
if device_comp:
    device = torch.device(device_comp)

torch.manual_seed(1337)
print(device)

encoded_train_data, encoded_test_data, encoder = create_data_encodings(path="./data/text.txt", encoder_config = "gpt2")
data_x, data_y = create_splits(encoded_train_data, encoded_test_data, mode='train')

model = GPTModel()
if compile:
    model = torch.compile(model).to(device)
    print("Compiled")
else:
    model = model.to(device)

print(model)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
for iter in range(1, max_iter):
    x, y = create_splits(encoded_train_data, encoded_test_data, mode='train')
    print(x.device, y.device)
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_iter == 0:
        fo = open("./data/output.txt", "a")
        loss_val = round(loss.item(), 4)
        current_step = iter
        output_current = encoder.decode(model.generate_captions(torch.zeros((1, 1), dtype = torch.long).to(device), 100)[0].tolist())
        print("Current Step: {}, Train Loss: {}".format(iter, loss_val))
        print("Current Output:")
        print(output_current)
        print(output_current, file = fo)
        print("-------------------------------------------------\n\n\n", file=fo)
        print("-------------------------------------------------\n\n\n")
        fo.close()

print("Training Ends")
print("Final loss: {}".format(loss.item()))
torch.save(model, "./checkpoint/shakespearModel")
print(encoder.decode(model.generate_captions(torch.zeros((1, 1), dtype = torch.long).to(device), 1000)[0].tolist()), file=fo)
