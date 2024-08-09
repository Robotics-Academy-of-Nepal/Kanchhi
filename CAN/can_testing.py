import torch
import torch.nn as nn
import spacy
import numpy as np
import random
import json
import pickle
import azure.cognitiveservices.speech as speechsdk
import requests
import uuid
from dotenv import load_dotenv
import os



load_dotenv()

translator_key = os.getenv('translator_key')
translator_endpoint = os.getenv('translator_endpoint')
location = os.getenv('translator_location')



# Load spaCy model and intents
nlp = spacy.load('en_core_web_sm')

with open('can.json', encoding='utf-8') as file:
    intents = json.load(file)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

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

model_path = 'best_chatbot.pth'
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
    if predicted_class in classes:
        return predicted_class
    else:
        return None

def get_response(predicted_intent, intents_json):
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == predicted_intent:
            return random.choice(intent['responses'])
            
    return None

def handle_user_input(user):
    predicted_intent = predict_class(user)
    if predicted_intent is None:
        return None
    response = get_response(predicted_intent, intents)
    return response


def speech_to_text():
    speech_config = speechsdk.SpeechConfig(
                subscription= os.getenv('speech_key'),
                region= os.getenv('speech_region')
            )

    speech_config.speech_recognition_language = "ne-NP"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )
    print("Listening.......")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        text = speech_recognition_result.text
        print(text)
        return text
    
def ne_to_en(text):
    if not all([translator_key,translator_endpoint,location]):
        raise ValueError("Please set the environment variables accurately.")

    path = '/translate'
    constructed_url = translator_endpoint + path

    params = {
        'api-version': '3.0',
        'from': 'ne',
        'to': ['en']
    }

    headers = {
        'Ocp-Apim-Subscription-Key': translator_key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceID': str(uuid.uuid4())
    }

    body = [{
        'text': text
    }]

    try:
        request = requests.post(constructed_url,params=params,headers=headers,json=body)
        response = request.json()
        translated_text = response[0]['translations'][0]['text']
        return translated_text
    except Exception as e:
        print("An error occured:", e)
byes = ["Bye.","bye","Bye"]
while True:
    user = speech_to_text()
    user = ne_to_en(user)
    print(user)
    for bye in byes:
        if user in bye:
            break
    output = handle_user_input(user)
    print(output)