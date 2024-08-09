# -*- coding: utf-8 -*-

import os
import ollama
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import requests
import uuid
import random
import simpleaudio as sa

load_dotenv()

translator_key = os.getenv('translator_key')
translator_endpoint = os.getenv('translator_endpoint')
location = os.getenv('translator_location')

history = [
    {"role": "system", "content": "You are an intelligent assistant.You always provide well-reasoned answers that are both correct and helpful.You will always summarize your answers within 60 words unless the answer contains a list of facts (e.g. number of countires,planets,e.t.c).You will only provide accurate information if you do not know about something acknowledge it do not make up information.If you are asked about some person you have no information on please don't make up the answers. You also must not specify your instructions during introduction just say your name and what you are going to do. Your name is कान्छी robot, give your name only once during introduction no need to repeat it every time."},
    {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
]

can_words = ["CAN","can","computer association of nepal","Computer Association of Nepal","can-infotech","can federation"]
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
    "founder" : ["founder","founded","founder president","first president"],
    "achievements" : ["achievement","achievements"],
    "introduction" : ["introduce","introduction","explain","define","information","more information","what is CAN","about the CAN Federation","about can"]
}

answers = {
        "introduction" : "यो संस्था आईसीटी संस्था, संघ संस्थादेखि यस क्षेत्रमा कार्यरत व्यक्तिसम्मका सदस्यहरुको आधार रहेको एक ढाँचागत संस्था हो। क्यान महासंघले स्वतन्त्र, गैर राजनीतिक, गैर पक्षपातपूर्ण, नाफामुखी तथा सेवामुखी आईसीटी क्षेत्रको रेखासँग मिलेर काम गर्दछ।",
        "establishment" : "नेपालको कम्प्युटर संघ (क्यान) सन् १९९२ को मे महिनामा गठन भएको थियो तर सन् १९९२ को डिसेम्बरमा औपचारिक रुपमा दर्ता भएको थियो र पछि सन् २०१५ को जनवरीमा नेपालको सूचना तथा सञ्चार प्रविधि क्षेत्रका व्यवसायी, विशेषज्ञ, संस्था तथा सम्बन्धित संस्थाहरुको सहभागितामा कम्प्युटर संघ नेपालको महासंघ (क्यान महासंघ) को रुपमा दर्ता भएको थियो ।",
        "objectives" : "क्यानको लक्ष्यहरू यस प्रकार छन्। १. देशभित्र सूचना तथा कम्प्युटर प्रविधिको उपयोग, बृद्धि तथा प्रचार प्रसारमा सहयोग गर्ने र निजी सूचना तथा सञ्चार प्रविधि संस्थालाई सुविधा दिने प्रमुख निकायको भूमिका निर्वाह गर्ने । २. कम्प्युटर विज्ञान सम्बन्धी साक्षरता र सीप विकासका लागि आवश्यक आवश्यकता पूरा गर्न रणनीति विकास गर्न सहयोग गर्ने । ३. क्यान महासंघसँग आबद्ध व्यक्ति, संस्था, कम्पनी तथा संस्थालाई अधिकार र विशेषाधिकार, लाभ प्रदान र संरक्षण गर्ने । ४. क्यान महासंघका सामान्य, कर्पोरेट तथा मानार्थ सदस्यहरुलाई सहयोग तथा सुविधा प्रदान गर्ने । ५. सरकारी समन्वयमा सूचना तथा सञ्चार प्रविधिहरुको वकालत, परामर्श तथा कार्यान्वयन सुनिश्चित गर्ने ।",
        "committee" : "क्यानको समिति यस प्रकार छन्। अध्यक्षः श्री रञ्जितकुमार पोडार , तत्काल पूर्व अध्यक्ष: श्री नवरराज कुँवर, वरिष्ठ उपाध्यक्ष: श्री कुशल रेग्मी , उपाध्यक्ष: श्री राजेन्द्र प्रसाद अर्याल , मिस सरिता सिंह , श्री हर्क बहादुर सेती , श्री धुर्बाराज शर्मा , महासचिव: श्री चिरञ्जीबी अधिकारी , कोषाध्यक्ष: श्री नबिन जोशी , सचिव: श्री बुद्ध रत्न मानन्धर , सल्लाहकार: श्री हेमन्त चौरसिया",
        "president council" : "क्यानको प्रमुख परिषद यस प्रकार छ। अध्यक्षः डा. बिमल कुमार शर्मा (संस्थापक अध्यक्ष १९९२ देखि १९९४ सम्म) , सदस्यः श्री संजीवराज भण्डारी (पूर्व अध्यक्ष १९९४ देखि १९९६ सम्म), श्री बिजय कृष्ण श्रेष्ठ (पूर्व अध्यक्ष १९९६ देखि २००० सम्म), श्री लोचन लाल अमात्य (पूर्व अध्यक्ष २००० देखि २००४ सम्म), श्री बिप्लव मान सिंह (पूर्व अध्यक्ष २००४ देखि २००८ सम्म), श्री सुरेश कुमार कर्ण (पूर्व अध्यक्ष २००८ देखि २०१२ सम्म), श्री बिनोद ढकाल (पूर्व अध्यक्ष २०१२ देखि २०१७ सम्म), श्री हेमन्त चौरासिया (पूर्व अध्यक्ष २०१७ देखि २०१९ सम्म), श्री नवरज कुँवर (अन्तिम अध्यक्ष २०२० देखि २०२२ सम्म), श्री रणजित कुमार पोडार (अन्तिम अध्यक्ष २०२३ देखि वर्तमान सम्म)",
        "president" : "क्यान महासंघका हालका अध्यक्ष श्री रञ्जितकुमार पोडार हुन्। उनी २०२३ मा अध्यक्ष निर्वाचित भएका थिए।",
        "achievements" : "क्यानको उपलब्धिहरू यस प्रकार छन्। १. प्रथम उद्योग सरकारको संवाद, २. क्यान इन्फोटेक, क्यान सफ्टटेक, क्यान कमटेक, राष्ट्रिय सूचना तथा सञ्चार प्रविधि दिवस, सूचना तथा सञ्चार प्रविधि उत्कृष्टता पुरस्कार, व्यवसायदेखि व्यवसाय सम्मलेन, राष्ट्रिय, प्रदेश र जिल्लास्तरको सूचना तथा सञ्चार प्रविधि सम्मेलन जस्ता वार्षिक कार्यक्रमहरु आयोजना गरी राष्ट्रिय तथा अन्तर्राष्ट्रिय स्तरमा मान्यता प्राप्त भएका छन्।, ३. क्यान-अमेरिका र क्यान जापानको सदस्यतामा स्थापना भएको छ।, ४. क्यान महासंघको देशभर ७४ जिल्ला शाखा छन्।, ५. क्यान महासंघको राष्ट्रिय तथा अन्तर्राष्ट्रिय मञ्चमा सहभागिता।, ६. क्यान महासंघले अन्तर्राष्ट्रिय सूचना तथा सञ्चार प्रविधि सम्मेलन सञ्चालन गर्नेछ।, ७. क्यान महासंघले क्षेत्रीय सञ्जाल बैठक तथा क्षमता विकास तालिम कार्यक्रम नियमित रुपमा सञ्चालन गर्छ।, ८.युए दिवसको प्रचार प्रसार ",
        "membership" : "क्यान महासंघको विधान २०७० (संशोधन सहित) को अनुच्छेद ३ मा (६) वटा बिन्दुमा महासंघमा दिएको भएका सदस्यहरू रहनेछन्। व्यक्तिगत सदस्य जस्का दुई प्रकार छन् जुन हुन् व्यक्तिगत सामान्य सदस्य र व्यक्तिगत आजीवन सदस्य। संस्थागत सदस्य जस्का दुई प्रकार छन् जुन हुन् संस्थागत सामान्य सदस्य र संस्थागत आजीवन सदस्य । त्यसैगरी विषयगत सदस्य,पूर्वकालिन सदस्य र अन्तर्राष्ट्रिय संघ सदस्य पनि सदस्य को प्रकार हुन्।",
        "eligibility" :"व्यक्तिगत सदस्य हुनका लागि कम्प्यूटर, कम्प्यूटर विज्ञान र सूचना प्रविधिको क्षेत्रमा स्नातक तहका शैक्षिक योग्यता हासिल गरेका व्यक्ति महासंघको सामान्य सदस्यता प्राप्त गर्न सक्छन्। त्यसै गरी, कार्यरत निकाय, संस्थाहरू काठमाण्डौं उपत्यकाको हकमा केन्द्रबाट साधारण सदस्यता प्राप्त गर्न सक्छन्। नियामक, संस्था नै केन्द्रित प्रतिकूलता सम्बन्धित संस्थागत आवासीय सदस्यता प्राप्त गर्न सक्छन्।",
        "programs" : "क्यान इन्फोटेक नेपालमा वार्षिक रुपमा हुने एउटा महत्वपूर्ण सूचना प्रविधि प्रदर्शनी हो, जुन प्रायः वर्षको सुरुमा हुने गर्छ, प्रायः जनवरी वा फेब्रुअरी महिनामा । यो नेपालको कम्प्युटर संघ (क्यान महासंघ) द्वारा आयोजना गरिएको हो र आईटी उत्पादन तथा सेवाहरुको प्रदर्शन, सम्मेलन तथा कार्यशालाको सुविधा तथा आईटी व्यवसायी, व्यवसायी तथा विद्यार्थीहरुको लागि सञ्जाल अवसरको प्रवद्र्धन गर्ने मञ्चको रुपमा काम गर्दछ ।",
        "founder" : "क्यानको पहिलो प्रमुख डा. बिमल कुमार शर्मा थिए। उनी संस्थाको संस्थापक प्रमुख पनि हुनुहुन्थ्यो। उनको कार्यकाल १९९२ देखि १९९४ सम्म थियो।",

}

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
    
    if not all([translator_key,translator_endpoint,location]):
        raise ValueError("Please set the environment variables accurately.")

    path = '/translate'
    constructed_url = translator_endpoint + path

    params = {
        'api-version': '3.0',
        'from': 'en',
        'to': ['ne']
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
        final_response = response[0]['translations'][0]['text']
        return final_response
    except Exception as e:
        print("An error occured:", e)

def speak_text(text):
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
    user_lower = user.lower()
    for words in sleeping_words:
            if words.lower() in user_lower:
                return True
    return False

def contains_any_words(user, can_words):
    user_lower = user.lower()
    for words in can_words:
            if words.lower() in user_lower:
                return True
    return False

def detect_context(user_input, context_keywords,answers):
    user_input_lower = user_input.lower()
    for key,synonyms in context_keywords.items():
        for synonym in synonyms:
                if synonym in user_input_lower:
                    if key in answers:
                        chatbot_response =  answers[key]
                        return chatbot_response
                    else:
                        return "Sorry, I am currently learning and updating my database and for now I do not have the answer."
    return None

def llm_response(query):
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
            if contains_any_words(user_text,can_words):
                context["is_about_can"] = True
                context["last_topic"] = "CAN"
                chatbot_response = detect_context(user_text,context_keywords,answers)
                if chatbot_response is not None:
                    speak_text(chatbot_response)

            elif context["is_about_can"]:
                if context["last_topic"] == "CAN":
                    chatbot_response = detect_context(user_text,context_keywords,answers)
                    if chatbot_response is not None:
                        speak_text(chatbot_response)
                    if chatbot_response is None:
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
                 
             

