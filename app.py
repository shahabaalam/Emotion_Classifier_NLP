import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./emotion-detection-model", local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained("./emotion-detection-model", local_files_only=True)
model.eval()

# Define emotion labels
emotion_labels = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Streamlit page configuration
st.title("Emotion Detection App")
st.write("Enter the text you'd like to analyze for emotion.")

# Text input
user_input = st.text_area("Text Input", "")

if st.button("Analyze"):
    if user_input:
        # Encode the user input and send to model
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits

        # Convert logits to probabilities, then to class labels
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = probabilities.argmax().item()
        emotion = emotion_labels[predicted_class]

        st.write(f"Predicted Emotion: {emotion}")
    else:
        st.write("Please enter some text to analyze.")
