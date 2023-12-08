import torch
from transformers import BertModel, BertTokenizer
import torch.nn as nn

# Define the BERT model and tokenizer
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert = BertModel.from_pretrained(bert_model_name)

class RestaurantReviewsModel(nn.Module):
    def __init__(self, bert, num_categories, num_polarities):
        super(RestaurantReviewsModel, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.category_classifier = nn.Linear(bert.config.hidden_size, num_categories)
        self.polarity_classifier = nn.Linear(bert.config.hidden_size, num_polarities)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        category_logits = self.category_classifier(pooled_output)
        polarity_logits = self.polarity_classifier(pooled_output)
        return category_logits, polarity_logits

# Create the model
num_categories = 5  # Number of categories: food, service, ambience, polarity, miscellaneous
num_polarities = 3  # Number of polarities: positive, negative, neutral
model = RestaurantReviewsModel(bert, num_categories, num_polarities)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

