import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn import metrics
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer
import pandas as pd
from models import Bert_CRF
from utils import NerDataset, PadBatch, get_all_labels

BATCH_SIZE = 3
LR = 0.01
EPOCHS = 10
MODEL_NAME = 'bert-base-uncased'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        loss = model(sentences, masks, labels)

        losses += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        break
    print(f'Epoch: {e}, Loss: {losses / step}')

def validate(e, model, data_loader, device, idx2tag):
    model.eval()
    Y, Y_hat = [], []
    losses = 0
    step = 0
    print('validating...')
    with torch.no_grad():
        for batch in tqdm(data_loader):
            step += 1

            sentences, labels, masks = batch
            sentences = sentences.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            y_hat = model(sentences, masks)

            print(eval_metrics(labels, y_hat, idx2tag))

            loss = model(sentences, masks, labels)
            losses += loss.item()

            for j in y_hat:
                Y_hat.extend(j)

            valid = (masks == 1)
            y_orig = torch.masked_select(labels, valid)
            Y.append(y_orig.cpu())
        Y = torch.cat(Y, dim=0).numpy()
        Y_hat = np.array(Y_hat)
        acc = (Y_hat == Y).mean() * 100
        print(f"Epoch: {e}, Val Loss: {losses/step:.3f}, Val Acc: {acc}")
        return model, losses/step, acc

def test(model, data_loader, device):
    model.eval()
    print('testing...')
    Y, Y_hat = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            sentences, labels, masks = batch
            sentences = sentences.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            y_hat = model(sentences, masks)
            
            for j in y_hat:
                Y_hat.extend(j)
            
            mask = (masks == 1).cpu()
            y_orig = torch.masked_select(labels, mask)
            Y.append(y_orig)
        Y = torch.cat(Y, dim=0).numpy()
        return Y, Y_hat

def eval_metrics(labels, preds, idx2tag):
    labels = labels.cpu().numpy().tolist()
    preds = preds.cpu().numpy().tolist()

    labels_tag, preds_tag = convert_idx_to_tag(labels, preds, idx2tag)
    f1 = f1_score(labels_tag, preds_tag)
    return f1

def convert_idx_to_tag(labels, preds, idx2tag):
    special_token_set = {'<PAD>', '[CLS]', '[SEP]'}
    labels_tag, preds_tag = [], []
    for label_sent, pred_sent in zip(labels, preds):
        for label, pred in zip(label_sent, pred_sent):
            if label in special_token_set:
                continue
            labels_tag.append(idx2tag[label])
            preds_tag.append(idx2tag[pred])
    return labels_tag, preds_tag
            


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    tag2idx, idx2tag = get_all_labels('conll/bio_type.txt')
    print(idx2tag)
    train_set = NerDataset('conll/valid.txt', tag2idx, tokenizer)
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, collate_fn=PadBatch)
    valid_set = NerDataset('conll/valid.txt', tag2idx, tokenizer)
    valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, collate_fn=PadBatch)
    test_set = NerDataset('conll/test.txt', tag2idx, tokenizer)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, collate_fn=PadBatch)
    model = Bert_CRF(n_classes=len(idx2tag), model_name=MODEL_NAME)

    optimizer  = AdamW(model.parameters(), lr=LR, eps=1e-6)
    total_steps = (len(train_set) // BATCH_SIZE) * EPOCHS if len(train_set) % BATCH_SIZE == 0 else (len(train_set) // BATCH_SIZE + 1) * EPOCHS
    warm_up_ratio = 0.1 # Define 10% steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for e in range(EPOCHS):
        train(e, model, train_loader, optimizer, scheduler, DEVICE)
        cand_mdoel, loss, acc = validate(e, model, valid_loader, DEVICE, idx2tag)

    Y, Y_hat = test(model, test_loader, DEVICE)
    y_true = [idx2tag[i] for i in Y]
    y_pred = [idx2tag[i] for i in Y_hat]


