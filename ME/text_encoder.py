# text_encoder.py
from transformers import DistilBertModel, DistilBertTokenizer
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    def forward(self, texts):
        # texts is expected to be a list of strings
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Using mean of last hidden state as representation
