import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pickle
import numpy as np
from gtts import gTTS
import os
import random
import json

from keras.models import load_model
model = load_model('Chatbot/chatbot_model.h5', compile=False)

from googletrans import Translator
import pyttsx3

# pygame for audio playback (optional, safe init)
from pygame import mixer

# Load intents, words, and classes
intents = json.loads(open('Chatbot/parkinson.json', encoding='utf-8').read())
words = pickle.load(open('Chatbot/words.pkl','rb'))
classes = pickle.load(open('Chatbot/classes.pkl','rb'))

# -----------------------------
# Helper Functions
# -----------------------------

def safe_init_mixer():
    """Initialize pygame mixer safely."""
    try:
        if not mixer.get_init():
            mixer.init()
    except Exception as e:
        print(f"[WARNING] Audio init skipped: {e}")

def clean_up_sentence(sentence, source_lang='en', target_lang='en'):
    translator = Translator()
    translation = translator.translate(sentence, src=source_lang, dest=target_lang).text
    sentence_words = nltk.word_tokenize(translation)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    print("This is the input text:", sentence_words)
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag:", w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json, source_lang='en', target_lang='en', 
                save_path="E:/2025 - 2026/DL/brain done/ITPDL11(Done)/Deploy/Project/Responses/response.mp3"):
    if not ints:
        return "Sorry, I didn't understand that."

    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    translator = Translator()
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            translation = translator.translate(result, src=source_lang, dest=target_lang).text
            result = translation

            # Generate TTS audio
            try:
                tts = gTTS(text=result, lang=target_lang)
                tts.save(save_path)
                safe_init_mixer()  # Initialize mixer safely
                sound = mixer.Sound(save_path)
                sound.play()
            except Exception as e:
                print(f"[WARNING] Audio playback skipped: {e}")

            return result
    
    return "You must ask the right questions."

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res
