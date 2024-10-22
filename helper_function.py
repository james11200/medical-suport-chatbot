import nltk
import pickle
import json
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

model = load_model('chatbot_model_lstm.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

WordNetLemmatizer = WordNetLemmatizer()

def GetCleanSentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [WordNetLemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def getWordsBag(sentence, words, show_details=True):
    sentence_words = GetCleanSentence(sentence)
    words_bag = [0] * len(words)
    for sentence in sentence_words:
        for i, word in enumerate(words):
            if word == sentence:
                words_bag[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    return np.array(words_bag)

def PredictClass(sentence, model):
    words_bag = getWordsBag(sentence, words, show_details=False)
    model_result = model.predict(np.array([words_bag]))[0]
    ERROR_THRESHOLD = 0.10  # Lowering the threshold
    results = [[i, result] for i, result in enumerate(model_result) if result > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        print(f"Intent: {classes[r[0]]}, Probability: {r[1]}")  # Debug print
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    if not return_list:
        print("No intent matched, returning default intent.")  # Debug print
    return return_list

def getResponse(intents, intents_json):
    if not intents:
        return "Sorry, I don't understand that."
    
    tag = intents[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def GetChatbotResponse(text):
    intent = PredictClass(text, model)
    results = getResponse(intent, intents)
    return results
