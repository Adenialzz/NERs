import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn import metrics
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer
import pandas as pd
from models import Bert_CRF
from utils import NerDataset, PadBatch, get_all_labels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--use_lstm', action='store_true')
    parser.add_argument('--use_crf', action='store_true')
    cfg = parser.parse_args()
    return cfg

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
        # break
    print(f'Epoch: {e}, Loss: {losses / step}')

def validate(e, model, data_loader, device, idx2tag):
    model.eval()
    losses = 0
    step = 0
    print('validating...')
    all_pred_tags, all_label_tags = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            step += 1

            sentences, labels, masks = batch
            sentences = sentences.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            y_hat = model(sentences, masks)

            loss = model(sentences, masks, labels)
            losses += loss.item()

            # print(f'labels: {labels[0]}')
            # print(f'preds: {y_hat[0]}')
            labels_tag, preds_tag = convert_idx_to_tag(labels, y_hat, idx2tag)
            all_label_tags.extend(labels_tag)
            all_pred_tags.extend(preds_tag)

        f1 = f1_score(all_label_tags, all_pred_tags)
        print(f"Epoch: {e}, Val Loss: {losses/step:.3f}, Val F1: {f1:.2f}")
        return model, losses/step, f1


def convert_idx_to_tag(labels, preds, idx2tag):
    special_token_set = {9, 10, 11} # {'<PAD>', '[CLS]', '[SEP]'}

    labels = labels.cpu().numpy().tolist()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy().tolist()
    labels_tag, preds_tag = [], []
    for label_sent, pred_sent in zip(labels, preds):
        label_tag_sent, pred_tag_sent = [], []
        for label, pred in zip(label_sent, pred_sent):
            if label in special_token_set:
                continue
            label_tag_sent.append(idx2tag[label])
            pred_tag_sent.append(idx2tag[pred])
        labels_tag.append(label_tag_sent)
        preds_tag.append(pred_tag_sent)
    return labels_tag, preds_tag

def main(cfg):
    tokenizer = BertTokenizer.from_pretrained(cfg.model_name)
    tag2idx, idx2tag = get_all_labels('conll/bio_type.txt')
    train_set = NerDataset('conll/train.txt', tag2idx, tokenizer)
    train_loader = DataLoader(dataset=train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=10, collate_fn=PadBatch)
    valid_set = NerDataset('conll/valid.txt', tag2idx, tokenizer)
    valid_loader = DataLoader(dataset=valid_set, batch_size=cfg.batch_size, shuffle=True, num_workers=10, collate_fn=PadBatch)
    test_set = NerDataset('conll/test.txt', tag2idx, tokenizer)
    test_loader = DataLoader(dataset=test_set, batch_size=cfg.batch_size, shuffle=True, num_workers=10, collate_fn=PadBatch)
    model = Bert_CRF(n_classes=len(idx2tag), model_name=cfg.model_name, use_crf=cfg.use_crf, use_lstm=cfg.use_lstm)

    optimizer  = AdamW(model.parameters(), lr=cfg.lr, eps=1e-6)
    total_steps = (len(train_set) // cfg.batch_size) * cfg.epochs if len(train_set) % cfg.batch_size == 0 else (len(train_set) // cfg.batch_size + 1) * cfg.epochs
    warm_up_ratio = 0.1 # Define 10% steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)

    for e in range(cfg.epochs):
        train(e, model, train_loader, optimizer, scheduler, cfg.device)
        cand_mdoel, loss, acc = validate(e, model, valid_loader, cfg.device, idx2tag)

            
if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
