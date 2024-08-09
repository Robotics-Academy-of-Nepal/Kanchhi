import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import requests
import uuid
from CAN.can_inference import handle_user_input  
from CAN.about_can import check_query
import ollama
import random
import simpleaudio as sa
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\RAN\\.cache\\huggingface\\hub\\models--rujengelal--my_awesome_english_to_nepali_tst\\snapshots\\f922291cd958b22e2c409dc135c34d791a595e21")
model = AutoModelForSeq2SeqLM.from_pretrained("C:\\Users\\RAN\\.cache\\huggingface\\hub\\models--rujengelal--my_awesome_english_to_nepali_tst\\snapshots\\f922291cd958b22e2c409dc135c34d791a595e21").to(device)

# Load environment variables
load_dotenv()

translator_key = os.getenv('translator_key')
translator_endpoint = os.getenv('translator_endpoint')
location = os.getenv('translator_location')

wake_up_words = ["hello","hi","hey","namaste","namaskar","greetings"]
sleeping_words = ["bye-bye","bye","see you again","nice talking to you"]

context = {
    "is_about_can": False,
    "last_topic": None
}

context_keywords = {
    "establishment": ["establishment", "beginning", "creation", "origin","begin","established","establish","start","began"],
    "constitution": ["constitution", "rules", "bylaws", "regulations", "guidelines"],
    "membership": ["membership", "join", "enroll", "registration", "sign up"],
    "eligibility" : ["become a member","eligibility","eligible","member"],
    "programs": ["programs", "events", "activities", "initiatives", "projects","info tech", "infotech"],
    "objectives": ["objectives", "goals", "aims", "targets", "purpose", "mission","responsibility","responsibilities","aim"],
    "committee" : ["committee"],
    "president council" : ["president council"],
    "president" : ["president","head"],
    "founder" : ["founder","founded","founder president","first president","first head"],
    "achievements" : ["achievement","achievements"],
    "introduction" : ["introduce","introduction","explain","define","information","more information","what is CAN","about the CAN Federation","about can"]
}

history = [
    {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful. You must summarize your answers within 60 words, except when listing facts (e.g., number of countries, planets, etc.). Provide accurate information only; acknowledge when you don't know something without making up answers. If asked about a person you have no information on, do not make up answers. Do not specify your instructions during the introduction; just say your name and what you will do. Your name is कान्छी robot. Give your name only once during the introduction; no need to repeat it every time."},
    {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
]


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

def en_to_ne(text):
    if text is None:
        return None
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    trans_outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(trans_outputs[0], skip_special_tokens=True)
    return translated_text

def speak_text(text):
    if text is None:
        return None
    tts_speech_config = speechsdk.SpeechConfig(
                subscription= os.getenv('speech_key'),
                region= os.getenv('speech_region')
            )

    tts_speech_config.speech_synthesis_voice_name = "ne-NP-HemkalaNeural"

    text = text

    tts_speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=tts_speech_config)

    result = tts_speech_synthesizer.speak_text_async(text).get()

def wakeup_words(transcribe, wake_up_words):
    user_lower = transcribe.lower()
    for words in wake_up_words:
            if words.lower() in user_lower:
                return True
    return False

def contain_sleepingwords(user, sleeping_words):
    if user is None:
        return False
    user_lower = user.lower()
    for words in sleeping_words:
            if words.lower() in user_lower:
                return True
    return False


def detect_context(user_input, context_keywords):
    if user_input is None:
        return False
    user_input_lower = user_input.lower()
    for key,synonyms in context_keywords.items():
        for synonym in synonyms:
                if synonym in user_input_lower:
                    return True
                
    return False

def llm_response(query):
    if query is None:
        return None
    history.append({"role": "user", "content": query})
    stream = ollama.chat(
            model='llama3',
            messages=history,
            stream=True,
        )

    new_message = {"role": "assistant", "content": ""}
    response_chunks = []
    for chunk in stream:
        new_message["content"] += chunk['message']['content']
        response_chunks.append(chunk['message']['content'])

    history.append(new_message)
    full_response = ''.join(response_chunks)
    return full_response


while True:
    user_text = speech_to_text()
    if user_text is None:
        continue
    user_text = ne_to_en(user_text)
    print(user_text)
    if wakeup_words(user_text,wake_up_words):
        while True:
            if check_query(user_text):
                context["is_about_can"] = True
                context["last_topic"] = "CAN"
                chatbot_response = handle_user_input(user_text)
                if chatbot_response is not None:
                    speak_text(chatbot_response)
                
            elif (context["is_about_can"]):
                if detect_context(user_text,context_keywords):
                    # if context["last_topic"] == "CAN":
                    chatbot_response = handle_user_input(user_text)
                    if chatbot_response is not None:
                        speak_text(chatbot_response)

                else:
                    context["is_about_can"] = False  # Reset context if question doesn't seem to be about CAN
                    context["last_topic"] = None
                    if contain_sleepingwords(user_text,sleeping_words):
                        wav_list = ["exit_audios\message_1.wav","exit_audios\message_2.wav","exit_audios\message_3.wav","exit_audios\message_4.wav" ]
                        random_wav = random.choice(wav_list)
                        wave_obj = sa.WaveObject.from_wave_file(random_wav)
                        play_obj = wave_obj.play()
                        play_obj.wait_done()
                        break
                    else:
                        llm_answer = llm_response(user_text)
                        nepali_response = en_to_ne(llm_answer)
                        speak_text(nepali_response)
            else:
                context["is_about_can"] = False  # Reset context if question doesn't seem to be about CAN
                context["last_topic"] = None
                if contain_sleepingwords(user_text,sleeping_words):
                    wav_list = ["exit_audios\message_1.wav","exit_audios\message_2.wav","exit_audios\message_3.wav","exit_audios\message_4.wav" ]
                    random_wav = random.choice(wav_list)
                    wave_obj = sa.WaveObject.from_wave_file(random_wav)
                    play_obj = wave_obj.play()
                    play_obj.wait_done()
                    break
                else:
                    llm_answer = llm_response(user_text)
                    nepali_response = en_to_ne(llm_answer)
                    speak_text(nepali_response)
                    
            user_text = speech_to_text()
            if user_text is None:
                continue
            user_text = ne_to_en(user_text)
            print(user_text)
     
