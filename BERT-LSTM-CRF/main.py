import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
import argparse
from tqdm import tqdm
import os
import os.path as osp
from seqeval.metrics import f1_score as cal_f1_seqeval
from sklearn.metrics import f1_score as cal_f1_sklearn
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from models import Bert_CRF
from utils import NerDataset, PadBatch, get_all_labels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--data_root', type=str, default='data/conll2003/')
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
    print(f'Epoch: {e}, Loss: {losses / step:.3f}')

def validate(e, model, data_loader, device, idx2tag):
    model.eval()
    losses = 0
    step = 0
    print('validating...')
    all_pred_tags, all_label_tags = [], []
    all_preds, all_labels = [], []
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

            if isinstance(y_hat, torch.Tensor):
                labels_ll, preds_ll = demask(masks, labels, y_hat)
            else:
                labels_ll, _ = demask(masks, labels, labels)
                preds_ll = y_hat
            all_labels.extend(labels_ll)
            all_preds.extend(preds_ll)

        all_pred_tags = [[idx2tag[idx] for idx in sent] for sent in all_preds]
        all_label_tags = [[idx2tag[idx] for idx in sent] for sent in all_labels]

        # for seqeval f1 calculating  [ [O, O, B-X, I-X, ...], [O, O, ...] ]  List[ List[ str]]
        f1_seqeval = cal_f1_seqeval(all_label_tags, all_pred_tags)
        # for sklearn f1 calculating  [ 1, 3, 2, 5, ... ] List[ int]
        f1_sklearn = cal_f1_sklearn([item for sent in all_labels for item in sent], [item for sent in all_preds for item in sent], labels=range(1, len(idx2tag)), average='weighted')

        print(f"Epoch: {e}, Val Loss: {losses/step:.3f}, Val F1(seqeval, sklearn): {f1_seqeval:.2f}, {f1_sklearn:.2f}")


def demask(mask, labels, preds):
    labels_ll, preds_ll = [], []
    for i, sent_mask in enumerate(mask):
        labels_sent, preds_sent = [], []
        for j, word_mask in enumerate(sent_mask):
            if word_mask == 0:
                continue
            labels_sent.append(labels[i][j].item())
            preds_sent.append(preds[i][j].item())
        labels_ll.append(labels_sent)
        preds_ll.append(preds_sent)
    return labels_ll, preds_ll


def main(cfg):
    tokenizer = BertTokenizer.from_pretrained(cfg.model_name)
    tag2idx, idx2tag = get_all_labels('data/conll/bio_type.txt')
    train_set = NerDataset(osp.join(cfg.data_root, 'train.txt'), tag2idx, tokenizer)
    train_loader = DataLoader(dataset=train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=PadBatch)
    valid_set = NerDataset(osp.join(cfg.data_root, 'valid.txt'), tag2idx, tokenizer)
    valid_loader = DataLoader(dataset=valid_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=PadBatch)
    test_set = NerDataset(osp.join(cfg.data_root, 'test.txt'), tag2idx, tokenizer)
    test_loader = DataLoader(dataset=test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=PadBatch)
    model = Bert_CRF(n_classes=len(idx2tag), model_name=cfg.model_name, use_crf=cfg.use_crf, use_lstm=cfg.use_lstm)

    optimizer  = AdamW(model.parameters(), lr=cfg.lr, eps=1e-6)
    total_steps = (len(train_set) // cfg.batch_size) * cfg.epochs if len(train_set) % cfg.batch_size == 0 else (len(train_set) // cfg.batch_size + 1) * cfg.epochs
    warm_up_ratio = 0.1 # Define 10% steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)

    for e in range(cfg.epochs):
        train(e, model, train_loader, optimizer, scheduler, cfg.device)
        validate(e, model, valid_loader, cfg.device, idx2tag)

            
if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
