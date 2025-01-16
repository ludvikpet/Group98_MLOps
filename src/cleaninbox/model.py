import torch.nn as nn 
from transformers import AutoModel

class BertTypeClassification(nn.Module):
    def __init__(self, model_name, num_classes):
        super(BertTypeClassification, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)

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

