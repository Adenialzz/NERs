import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

class NerDataset(Dataset):
    def __init__(self, f_path, tag2idx, tokenizer, max_len=256):
        self.tag2idx = tag2idx
        assert tag2idx['<PAD>'] == 0
        self.sents, self.labels = [], []
        self.max_len = max_len
        self.tokenizer = tokenizer
        fp = open(f_path, 'r')
        
        sentence, sent_labels = [], []
        for line in fp:
            lsp = line.strip().split(' ')
            word, label = lsp[0], lsp[-1]
            if line != '\n':
                sentence.append(word)
                sent_labels.append(label)
            else:
                self.sents.append(sentence)
                self.labels.append(sent_labels)
                sentence, sent_labels = [], []

    def __getitem__(self, idx):
        sentence, sent_labels = self.sents[idx], self.labels[idx]
        token_ids = self.tokenizer.encode(sentence, add_special_token=True, max_length=self.max_len + 2, truncation=True, return_tensors='pt')  # truncation
        label_ids = torch.tensor( [self.tag2idx['<PAD>']] + [self.tag2idx[label] for label in sent_labels] + [self.tag2idx['<PAD>']] )
        seqlen = len(label_ids)
        return token_ids, label_ids, seqlen

    def __len__(self):
        return len(self.sents)

def PadBatch(batch):
    tokens, labels, _ = zip(*batch)  # [item] * 128, shape: (1, seqlen), (seqlen, )
    token_tensors = pad_sequence([item.transpose(1, 0) for item in tokens], batch_first=True).squeeze()
    label_tensors = pad_sequence([item.unsqueeze(-1) for item in labels], batch_first=True).squeeze()   # <PAD> is at position 0
    
    mask = (token_tensors > 0)  # <PAD> is at position 0
    return token_tensors, label_tensors, mask

def get_all_labels(filename):
    ner_type = pd.read_csv(filename, sep=' ')
    ners = ner_type['postfix'].to_list()
    labels = ['<PAD>']
    for n in ners:
        labels.extend(["B-" + n, "I-" + n])
    labels.append('O')
    
    tag2idx = {tag: idx for idx, tag in enumerate(labels)}
    idx2tag = {idx: tag for idx, tag in enumerate(labels)}
    print(f'all type len is {len(labels)}')
    return tag2idx, idx2tag

def get_zh_labels():
    # <PAD> should be placed at position 0
    labels = ['<PAD>', 'B-CONT','B-EDU','B-LOC','B-NAME','B-ORG','B-PRO','B-RACE','B-TITLE',
                'I-CONT','I-EDU','I-LOC','I-NAME','I-ORG','I-PRO','I-RACE','I-TITLE',
                'O','S-NAME','S-ORG','S-RACE']
    tag2idx = {tag: idx for idx, tag in enumerate(labels)}
    idx2tag = {idx: tag for idx, tag in enumerate(labels)}
    print(f'all type len is {len(labels) - 1}')  # remove <PAD>
    return tag2idx, idx2tag

