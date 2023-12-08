from flask import Flask, render_template, request, jsonify
from transformers import BertModel, BertTokenizer
import torch
import joblib
import numpy as np
from restaurant_model import RestaurantReviewsModel

app = Flask(__name__)

# Define the paths to your model and label encoders
model_path = 'C:/Users/vsing/OneDrive/Desktop/Capstone Project/Capstone_project_Sentiment_Analysis_Aspect_Level/restaurant_reviews_model.pth'  # Replace with the path to your saved model
label_encoder_path = 'label_encoders.pkl'  # Replace with the path to your saved label encoders

# Define the BERT model and tokenizer
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert = BertModel.from_pretrained(bert_model_name)

num_categories = 5  
num_polarities = 3

# Load the saved model
model = RestaurantReviewsModel(bert, num_categories, num_polarities)  # Create an instance of your model
model.load_state_dict(torch.load(model_path))
model.eval()

# Load the saved label encoders
le_category = joblib.load(label_encoder_path + "_category")
le_polarity = joblib.load(label_encoder_path + "_polarity")

# Define your tokenizer and maximum text length
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_text_length = 128  # Define your desired maximum text length

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_text(text, tokenizer, max_length):
    inputs = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    return inputs

@app.route('/result', methods=['POST'])
def analyze():
    user_input = request.form['user_input']
    unseen_sentences = [user_input]  # Use the user's input

    # Preprocess the unseen sentences and tokenize them
    unseen_inputs = [preprocess_text(text, tokenizer, max_text_length) for text in unseen_sentences]

    # Prepare input tensors for the unseen data
    unseen_input_ids = torch.cat([inputs['input_ids'] for inputs in unseen_inputs], dim=0)
    unseen_attention_mask = torch.cat([inputs['attention_mask'] for inputs in unseen_inputs], dim=0)
    unseen_token_type_ids = torch.cat([inputs['token_type_ids'] for inputs in unseen_inputs], dim=0)

    # Use the loaded model to make predictions on the unseen data
    with torch.no_grad():
        category_logits, polarity_logits = model(unseen_input_ids, unseen_attention_mask, unseen_token_type_ids)

    # Convert the model's predictions to actual labels
    category_preds = le_category.inverse_transform(category_logits.argmax(1).cpu().numpy())
    polarity_preds = le_polarity.inverse_transform(polarity_logits.argmax(1).cpu().numpy())

    sentiment = polarity_preds[0]
    aspect = category_preds[0]

    return jsonify({'sentiment': sentiment, 'aspect': aspect})

if __name__ == '__main__':
    app.run(debug=True, port=10000)

