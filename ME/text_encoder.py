# text_encoder.py
from transformers import BertModel, BertTokenizer
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, texts):
        # texts is expected to be a list of strings
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.pooler_output  # Use the pooled output by default, or you can modify as needed
