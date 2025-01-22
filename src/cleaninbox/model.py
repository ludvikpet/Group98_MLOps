import torch.nn as nn 
import torch
from transformers import AutoModel, AutoTokenizer

class BertTypeClassification(nn.Module):
    def __init__(self, model_name, num_classes):
        super(BertTypeClassification, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # Use the pooled output ([CLS] token) for classification
        pooled_output = outputs.pooler_output

        # Pass through the classification head
        logits = self.classifier(pooled_output)

        return logits
    
    def decode_input_ids(self,input_ids):
        #decodes a batch of input ids to a sentence in list format. Used for plotting topk distribution during training.
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

def top_k_logits(logits, k):
    '''
    returns the top k logits (softmax) and their indices

    Args:
        logits: torch.tensor, shape (1, num_classes)
        k: int, number of top logits to return
    '''
    softmax = nn.Softmax(dim=1)
    softmax_logits = softmax(logits)
    topk_values, topk_indices = torch.topk(softmax_logits, k, dim=1)
    topk_values = topk_values.cpu().numpy().tolist()[0]
    topk_indices = topk_indices.cpu().numpy().tolist()[0]
    return topk_values, topk_indices
