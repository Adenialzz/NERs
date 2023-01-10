import torch
from torch.utils.data import Dataset
import pandas as pd

class NerDataset(Dataset):
    def __init__(self, f_path, tag2idx, tokenizer, inference_df=None):
        self.tag2idx = tag2idx
        self.sents = []
        self.tags_li = []
        self.max_len = 256
        self.tokenizer = tokenizer
        if inference_df is not None:
            data = inference_df
        else:
            data = pd.read_csv(f_path, sep=' ')
        data.fillna(value='O', inplace=True)

        tags = data['O'].to_list()
        words = data['-DOCSTART-'].to_list()
        word, tag = [], []
        for char, t in zip(words, tags):
            if char != '.':
                word.append(char)
                tag.append(t)
            else:
                if len(word) >= self.max_len - 2:
                    self.sents.append(['[CLS]'] + word[: self.max_len] + [char] + ['[SEP]'])
                    self.tags_li.append(['[CLS]'] +tag[: self.max_len] + [t] + ['[SEP]']) 
                else:
                    self.sents.append(['[CLS]'] + word + [char] + ['[SEP]'])
                    self.tags_li.append(['[CLS]'] + tag + [t] + ['[SEP]']) 
                word, tag = [], []
        if word:
            if len(word) >= self.max_len - 2:
                self.sents.append(['[CLS]'] + word[: self.max_len] + ['[SEP]'])
                self.tags_li.append(['[CLS]'] +tag[: self.max_len] + ['[SEP]']) 
            else:
                self.sents.append(['[CLS]'] + word + ['[SEP]'])
                self.tags_li.append(['[CLS]'] + tag + ['[SEP]']) 
            word, tag = [], []


    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
        token_ids = self.tokenizer.convert_tokens_to_ids(words)
        label_ids = [self.tag2idx[tag] for tag in tags]
        seqlen = len(label_ids)
        return token_ids, label_ids, seqlen

    def __len__(self):
        return len(self.sents)

def PadBatch(batch):
    maxlen = max(i[2] for i in batch)
    token_tensors = torch.LongTensor([i[0] + [0] * (maxlen - len(i[1])) for i in batch])
    label_tensors = torch.LongTensor([i[1] + [9] * (maxlen - len(i[1])) for i in batch])
    mask = (token_tensors > 0)
    return token_tensors, label_tensors, mask

def get_all_labels(filename):
    ner_type = pd.read_csv(filename, sep=' ')
    ners = ner_type['postfix'].to_list()
    labels = []
    for n in ners:
        labels.extend(["B-" + n, "I-" + n])
    labels.append('O')
    labels.extend(['<PAD>', '[CLS]', '[SEP]'])
    
    tag2idx = {tag: idx for idx, tag in enumerate(labels)}
    idx2tag = {idx: tag for idx, tag in enumerate(labels)}
    print(f'all type len is {len(labels)}')
    return tag2idx, idx2tag
