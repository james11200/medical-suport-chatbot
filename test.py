import json
import pickle
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Load the trained LSTM model
model = load_model('chatbot_model_lstm.h5')

# Load the intents JSON file and necessary data
with open('intents.json', 'r') as file:
    intents = json.load(file)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Initialize WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
max_word_bag_length = 88
def clean_sentence(sentence):
    # Tokenize the sentence and lemmatize each word
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [wordnet_lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def get_words_bag(sentence, words, max_word_bag_length, show_details=False):
    sentence_words = clean_sentence(sentence)
    words_bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                words_bag[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    # Ensure words_bag has the correct length before reshaping
    if len(words_bag) < max_word_bag_length:
        words_bag += [0] * (max_word_bag_length - len(words_bag))
    elif len(words_bag) > max_word_bag_length:
        words_bag = words_bag[:max_word_bag_length]
    # Reshape words_bag to match the input shape of the LSTM model
    words_bag = np.array(words_bag).reshape(1, max_word_bag_length, 1)  # <-- Ensure correct reshaping
    return words_bag




def predict_class(sentence, model):
    words_bag = get_words_bag(sentence, words, max_word_bag_length, show_details=False)

    model_result = model.predict(np.array([words_bag]))[0]
    error_threshold = 0.25
    results = [[i, result] for i, result in enumerate(model_result) if result > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_json, tag):
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def get_chatbot_response(text):
    intent = predict_class(text, model)
    tag = intent[0]['intent']
    response = get_response(intents, tag)
    return response

# Example usage
text_input = "Hi there"
response = get_chatbot_response(text_input)
print(response)
