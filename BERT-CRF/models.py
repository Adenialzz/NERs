import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torchcrf import CRF

class Bert_CRF(nn.Module):
    def __init__(self, n_classes, model_name='bert-base-uncased', embedding_dim=768, hidden_dim=256, use_crf=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.bert = BertModel.from_pretrained(model_name)
        for _, param in self.bert.named_parameters():
            param.requires_grad = False
        self.use_crf = use_crf
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(embedding_dim, n_classes)
        if self.use_crf:
            self.crf = CRF(n_classes, batch_first=True)
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def _get_features(self, sentence):
        with torch.no_grad():
            outputs = self.bert(sentence)
        enc = outputs.last_hidden_state
        enc = self.dropout(enc)
        logits = self.linear(enc)
        return logits

    def forward(self, sentence, mask, label=None):
        logits = self._get_features(sentence)
        if label is not None:
            if self.use_crf:
                loss = -self.crf.forward(logits, label, mask, reduction='mean')
            else:
                loss = self.loss_func(logits.reshape(-1, logits.shape[-1]), label.reshape(-1))
            return loss
        else:
            if self.use_crf:
                result = self.crf.decode(logits, mask)
                return result
            else:
                return torch.argmax(logits, dim=-1)

