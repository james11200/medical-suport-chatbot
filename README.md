
# Medical Support Chatbot

This project is the final project of an Information Retrieval course. It involves scraping Reddit comments related to medical topics to build a Medical Assistance Chatbot with Intent Classification. The chatbot utilizes Natural Language Processing (NLP) techniques and a Long Short-Term Memory (LSTM) model for intent recognition, enabling it to understand and classify user inquiries effectively. Implemented in Python, it provides medical responses based on user queries, guiding users through various medical topics and support options.



## Features
- Scrapes medical data from Reddit for training.
- Uses an LSTM model for classifying intents from user input.
- Deployed using a simple graphical interface built with Tkinter.
- Can be easily expanded with additional intents and responses.

## Project Structure
```
├── chatbot.py             # Main chatbot interface with Tkinter
├── helper_function.py     # Helper functions for chatbot logic
├── intents.json           # Intents data in JSON format
├── test.py                # Script to test the chatbot without the GUI
├── train_model.py         # Script to train the LSTM model
├── words.pkl              # Saved vocabulary for the chatbot
├── classes.pkl            # Saved classes for the chatbot
└── chatbot_model_lstm.h5  # Trained LSTM model
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/medical-support-chatbot.git
   cd medical-support-chatbot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the `train_model.py` script to train the chatbot:
   ```
   python train_model.py
   ```

4. Start the chatbot interface:
   ```
   python chatbot.py
   ```

## Usage
- Run the chatbot_app.py program.
- Type your query into the chatbox and click the send button, and the bot will respond based on the trained model.
- You can modify the `intents.json` file to add new intents and responses.

## Intents Structure
The `intents.json` file contains intents in the following structure:
```json
{
    "intents": [
        {"tag": "medication_info",
        "patterns": [
                "What is ibuprofen used for?",
                "Tell me about aspirin.",
                "What are the side effects of metformin?",
                "Is it safe to take Tylenol with alcohol?",
                "How should I take amoxicillin?"
        ],
        "responses": [
                "Ibuprofen is commonly used to relieve pain and reduce inflammation.",
                "Aspirin is used to reduce fever, pain, and inflammation; it can also help prevent heart attacks.",
                "Common side effects of metformin include nausea, vomiting, and stomach upset.",
                "It is generally advised to avoid alcohol while taking Tylenol due to the risk of liver damage.",
                "Amoxicillin should be taken as prescribed, usually with a full glass of water."
        ]
        },
        ...
    ]
}
```
Each intent has a `tag`, a list of `patterns` (user inputs), and corresponding `responses`.
