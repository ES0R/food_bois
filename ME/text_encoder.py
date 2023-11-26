from transformers import DistilBertModel, DistilBertTokenizer
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    def forward(self, inputs):
        # Here, inputs should be a dictionary containing 'input_ids', 'attention_mask', etc.
        outputs = self.model(**inputs)
        return outputs[0].mean(dim=1)


# from transformers import BertModel, BertTokenizer
# import torch.nn as nn
# class TextEncoder(nn.Module):
#     def __init__(self):
#         super(TextEncoder, self).__init__()
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         self.model = BertModel.from_pretrained('bert-base-uncased')

#     def forward(self, inputs):
#         Here, inputs should be a dictionary containing 'input_ids', 'attention_mask', etc.
#         outputs = self.model(**inputs)
#         return outputs.last_hidden_state.mean(dim=1)  # Using mean of last hidden state as representation