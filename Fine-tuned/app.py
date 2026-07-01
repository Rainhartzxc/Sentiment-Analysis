import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import numpy as np
import warnings

# Suppress the specific warning about sentencepiece
warnings.filterwarnings("ignore", message="The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option which is not implemented in the fast tokenizers.")

# --- CONFIGURATION ---
MODEL_PATH = "./sentiment_deberta_model/"
MIN_WORDS = 3

# --- MODEL LOADING ---
# Use Streamlit's caching to load the model only once, improving performance.
@st.cache_resource
def load_model_assets():
    """
    Loads the fine-tuned model, tokenizer, and label encoder from the local directory.
    The `@st.cache_resource` decorator ensures this function is only run once.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        label_encoder = joblib.load(MODEL_PATH + "label_encoder.joblib")
        # Ensure the model is in evaluation mode
        model.eval()
        return tokenizer, model, label_encoder
    except Exception as e:
        # If loading fails, display an informative error message in the app.
        st.error(f"Error loading model assets: {e}")
        st.error(f"Please make sure the '{MODEL_PATH}' folder exists and contains all necessary model files.")
        return None, None, None

# --- PREDICTION FUNCTION ---
def predict_sentiment(text, tokenizer, model, label_encoder):
    """
    Predicts the sentiment of a given text using the loaded model.
    
    Args:
        text (str): The input text from the user.
        tokenizer: The loaded tokenizer.
        model: The loaded fine-tuned model.
        label_encoder: The loaded label encoder.
        
    Returns:
        tuple: A tuple containing the predicted label (str) and the confidence score (float).
    """
    # 1. Tokenize the input text
    inputs = tokenizer(text,
                       return_tensors="pt",
                       truncation=True,
                       padding=True,
                       max_length=512) # Use a safe max_length

    # 2. Make a prediction (no gradient calculation needed)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # 3. Get probabilities and the predicted class index
    probabilities = torch.softmax(logits, dim=1).flatten()
    predicted_class_index = torch.argmax(probabilities).item()
    
    # 4. Get the confidence score for the predicted class
    confidence = probabilities[predicted_class_index].item()
    
    # 5. Decode the predicted class index back to its string label
    predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
    
    return predicted_label, confidence


st.set_page_config(page_title="Student Feedback Sentiment Analysis", layout="wide")
st.title("🎓 Student Feedback Sentiment Analysis")
st.markdown("Enter student feedback text below. The model will predict whether the sentiment is **Positive**, **Negative**, or **Neutral**.")
st.markdown(f"**Rule:** Feedback with fewer than `{MIN_WORDS}` words will be classified as 'Lack of Context'.")

# Load model assets and handle potential errors
tokenizer, model, label_encoder = load_model_assets()

if tokenizer and model and label_encoder:
    # Create a text area for user input
    user_input = st.text_area("Enter student feedback here:", height=150, placeholder="e.g., 'The professors are very helpful and the library has great resources.'")

    # Create a button to trigger the analysis
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            # Check the business rule: word count
            word_count = len(user_input.strip().split())
            
            if word_count < MIN_WORDS:
                st.warning(f"⚠️ **Lack of Context** - The input has only {word_count} word(s). Please provide more detail.")
            else:
                # If rule is passed, make a prediction
                with st.spinner('Analyzing...'):
                    predicted_label, confidence = predict_sentiment(user_input, tokenizer, model, label_encoder)
                
                st.subheader("Analysis Result")
                
                # Display the result with a corresponding emoji and color
                if predicted_label == 'positive':
                    st.success(f"**Sentiment: Positive** 👍 ({confidence:.2%} confidence)")
                elif predicted_label == 'negative':
                    st.error(f"**Sentiment: Negative** 👎 ({confidence:.2%} confidence)")
                else: # neutral
                    st.info(f"**Sentiment: Neutral** 😐 ({confidence:.2%} confidence)")
        else:
            st.warning("Please enter some text to analyze.")
else:
    st.header("Application is not ready.")
    st.write("Could not load the machine learning model. Please check the terminal for error messages.")