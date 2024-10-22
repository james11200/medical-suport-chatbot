import nltk
import pickle
import random
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.optimizers import Adam

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Read intents file
intents = pd.read_json('intents.json')

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.']

# Loop through each sentence in the intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add to documents in the corpus
        documents.append((word_list, intent['tag']))
        # Add to classes if it's not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create a mapping of words to indices
word_to_index = {word: i for i, word in enumerate(words)}
vocab_size = len(words)  # Size of vocabulary based on unique words
embedding_dim = 100  # You can choose 50, 100, 200, etc.
max_sequence_length = 20  # Maximum length of sequences (set based on the dataset)

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create the training set, convert each sentence to a sequence of word indices
for doc in documents:
    sequence = [word_to_index[word] for word in doc[0] if word in word_to_index]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([sequence, output_row])

# Shuffle the features and convert to np.array
random.shuffle(training)
training = np.array(training, dtype=object)

# Create train and test lists
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

# Pad sequences to ensure consistent input length for LSTM
from keras.preprocessing.sequence import pad_sequences
train_x_padded = pad_sequences(train_x, maxlen=max_sequence_length)

# Define the model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
model.fit(train_x_padded, np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model_lstm.h5')
import nltk
import pickle
import random
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.optimizers import Adam

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Read intents file
intents = pd.read_json('intents.json')

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.']

# Loop through each sentence in the intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add to documents in the corpus
        documents.append((word_list, intent['tag']))
        # Add to classes if it's not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create a mapping of words to indices
word_to_index = {word: i for i, word in enumerate(words)}
vocab_size = len(words)  # Size of vocabulary based on unique words
embedding_dim = 100  # You can choose 50, 100, 200, etc.
max_sequence_length = 20  # Maximum length of sequences (set based on the dataset)

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create the training set, convert each sentence to a sequence of word indices
for doc in documents:
    sequence = [word_to_index[word] for word in doc[0] if word in word_to_index]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([sequence, output_row])

# Shuffle the features and convert to np.array
random.shuffle(training)
training = np.array(training, dtype=object)

# Create train and test lists
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

# Pad sequences to ensure consistent input length for LSTM
from keras.preprocessing.sequence import pad_sequences
train_x_padded = pad_sequences(train_x, maxlen=max_sequence_length)

# Define the model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
model.fit(train_x_padded, np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model_lstm.h5')
