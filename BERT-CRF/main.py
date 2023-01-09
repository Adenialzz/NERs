import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import warnings
import argparse
import numpy as np
from sklearn import metrics
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
from models import Bert_CRF
from utils import NerDataset, PadBatch
from tqdm import tqdm

BATCH_SIZE = 64
LR = 0.01
EPOCHS = 10
MODEL_NAME = 'bert-base-uncased'
DEVICE = 'cuda'

def train(e, model, dataloader, optimizer, scheduler, device):
    model.train()
    model = model.to(device)
    losses = 0.0
    step = 0
    print('training...')
    for batch in tqdm(dataloader):
        step += 1
        sentences, labels, masks = batch

        sentences = sentences.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        loss = model(sentences, labels, masks)

        losses += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    print(f'Epoch: {e}, Loss: {losses / step}')

if __name__ == '__main__':
    train_set = NerDataset('conll/train.txt', model_name=MODEL_NAME)
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, collate_fn=PadBatch)
    model = Bert_CRF(n_classes=len(train_set.labels), model_name=MODEL_NAME)

    optimizer  = AdamW(model.parameters(), lr=LR, eps=1e-6)
    total_steps = (len(train_set) // BATCH_SIZE) * EPOCHS if len(train_set) % BATCH_SIZE == 0 else (len(train_set) // BATCH_SIZE + 1) * EPOCHS
    warm_up_ratio = 0.1 # Define 10% steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for e in range(EPOCHS):
        train(e, model, train_loader, optimizer, scheduler, DEVICE)


