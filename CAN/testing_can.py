import torch
import torch.nn as nn
import spacy
import numpy as np
import random
import json
import pickle

# Load spaCy model and intents
nlp = spacy.load('en_core_web_sm')

with open('about_can.json', encoding='utf-8') as file:
    intents = json.load(file)
words = pickle.load(open('about_can_words.pkl', 'rb'))
classes = pickle.load(open('about_can_classes.pkl', 'rb'))

class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

model_path = 'about_can_best_chatbot.pth'
model = ChatbotModel(input_size=len(words), hidden_size=512, output_size=len(classes))
model.load_state_dict(torch.load(model_path))
model.eval()

def clean_up_sentence(sentence):
    doc = nlp(sentence)
    sentence_words = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    bow_tensor = torch.from_numpy(bow).float().unsqueeze(0)
    outputs = model(bow_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = classes[predicted.item()]
    return predicted_class

def is_query_about_can_organization(query):
    predicted_class = predict_class(query)
    # Check if the predicted class is 'CAN organization' or another tag indicating it
    return predicted_class == 'can_organization'

def check_query(query):
    return is_query_about_can_organization(query)

# Example usage
while True:
    user_input = input("Enter your query: ")
    if user_input == "bye":
        break
    
    result = is_query_about_can_organization(user_input)
    print("Is the query about CAN organization?", result)

