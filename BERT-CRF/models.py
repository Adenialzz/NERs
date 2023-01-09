import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torchcrf import CRF

class Bert_CRF(nn.Module):
    def __init__(self, n_classes, model_name='bert-base-uncased', embedding_dim=768, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(embedding_dim, n_classes)
        self.crf = CRF(n_classes, batch_first=True)

    def _get_features(self, sentence):
        with torch.no_grad():
            outputs = self.bert(sentence)
        enc = outputs.last_hidden_state
        enc = self.dropout(enc)
        feats = self.linear(enc)
        return feats

    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence)
        if not is_test:
            loss=-self.crf.forward(emissions, tags, mask, reduction='mean')
            return loss
        else:
            decode = self.crf.decode(emissions, mask)
            return decode

