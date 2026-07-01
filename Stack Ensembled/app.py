import os
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.sparse import hstack
import streamlit as st
import re

# ==============================================================================
# PREDICTION PIPELINE CLASS
# ==============================================================================
class SentimentPredictor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load all components
        self.tfidf_vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.joblib'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        self.svm_model = joblib.load(os.path.join(model_dir, 'svm_base_meta.joblib'))
        self.lgbm_model = joblib.load(os.path.join(model_dir, 'lgbm_base_meta.joblib'))
        self.stacking_model = joblib.load(os.path.join(model_dir, 'stacking_final_classifier.joblib'))
        self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
        
        # Load the 5-fold DeBERTa ensemble
        self.deberta_ensemble = []
        self.tokenizers = []
        num_folds = 5 
        for i in range(1, num_folds + 1):
            fold_dir = os.path.join(model_dir, f'fold_{i}')
            tokenizer = AutoTokenizer.from_pretrained(fold_dir)
            model = AutoModelForSequenceClassification.from_pretrained(fold_dir)
            model.to(self.device)
            model.eval()
            self.deberta_ensemble.append(model)
            self.tokenizers.append(tokenizer)

    def _get_ensemble_embeddings(self, text_list):
        all_fold_embeddings = []
        with torch.no_grad():
            for model, tokenizer in zip(self.deberta_ensemble, self.tokenizers):
                inputs = tokenizer(text_list, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
                base_model = getattr(model, model.base_model_prefix)
                outputs = base_model(**inputs)
                fold_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_fold_embeddings.append(fold_embedding)
        return np.mean(all_fold_embeddings, axis=0)

    def predict_proba(self, text):
        if isinstance(text, str):
            text_list = [text]
        else:
            raise ValueError("Input must be a single string.")

        # 1. Generate DeBERTa Embeddings
        bert_features = self._get_ensemble_embeddings(text_list)
        
        # 2. Generate TF-IDF Features
        tfidf_features = self.tfidf_vectorizer.transform(text_list)
        
        # 3. Combine and Scale Features
        combined_features = hstack([bert_features, tfidf_features]).toarray()
        scaled_features = self.scaler.transform(combined_features)
        
        # 4. Get predictions from base meta-models
        svm_preds = self.svm_model.predict_proba(scaled_features)
        lgbm_preds = self.lgbm_model.predict_proba(scaled_features)
        
        # 5. Create input for the stacking classifier
        stacked_input = np.concatenate([svm_preds, lgbm_preds], axis=1)
        
        # 6. Final probability prediction
        final_probabilities = self.stacking_model.predict_proba(stacked_input)[0]
        
        # 7. Decode the label and create a probability dictionary
        predicted_class_index = np.argmax(final_probabilities)
        predicted_label = self.label_encoder.inverse_transform([predicted_class_index])[0]
        
        prob_dict = {label: prob for label, prob in zip(self.label_encoder.classes_, final_probabilities)}
        
        return predicted_label, prob_dict

# ==============================================================================
# WORD IMPORTANCE VISUALIZATION
# This is a proxy method using TF-IDF scores to highlight influential words.
# ==============================================================================
def highlight_word_importance(text, predictor, predicted_label):
    # Get the fitted TF-IDF vectorizer from the predictor
    vectorizer = predictor.tfidf_vectorizer
    
    # Get TF-IDF scores for the input text
    tfidf_scores = vectorizer.transform([text]).toarray().flatten()
    feature_names = vectorizer.get_feature_names_out()
    word_score_map = {word: tfidf_scores[vectorizer.vocabulary_[word]] for word in vectorizer.vocabulary_ if word in text.lower().split()}

    # Normalize scores for color intensity
    max_score = max(word_score_map.values()) if word_score_map else 0
    
    # Define sentiment colors
    sentiment_colors = {
        'positive': (77, 175, 74),  # Green
        'negative': (228, 26, 28),   # Red
        'neutral':  (55, 126, 184)   # Blue
    }
    base_color_rgb = sentiment_colors.get(predicted_label, (128, 128, 128)) # Default gray

    highlighted_text = ""
    for word in re.split(r'(\s+)', text): # Split but keep spaces
        if word.strip().lower() in word_score_map:
            score = word_score_map[word.strip().lower()]
            # Opacity based on normalized score
            opacity = score / max_score if max_score > 0 else 0
            # Create a background color with varying opacity
            color = f"rgba({base_color_rgb[0]}, {base_color_rgb[1]}, {base_color_rgb[2]}, {opacity:.2f})"
            highlighted_text += f'<span style="background-color: {color}; border-radius: 5px; padding: 2px 4px;">{word}</span>'
        else:
            highlighted_text += word
            
    return highlighted_text

# ==============================================================================
# STREAMLIT APP
# ==============================================================================

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Feedback Sentiment Analysis",
    page_icon="🎓",
    layout="centered"
)

# --- Model Loading ---
# Use Streamlit's caching to load the model only once
@st.cache_resource
def load_model():
    model_dir = 'student_feedback_model_ensemble_B1_G3'
    if not os.path.exists(model_dir):
        st.error(f"Model directory not found! Please make sure the '{model_dir}' folder is in the same directory as app.py.")
        return None
    try:
        predictor = SentimentPredictor(model_dir)
        return predictor
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

predictor = load_model()

# --- App UI ---
st.title("🎓 Student Feedback Sentiment Analysis")
st.markdown("This system analyzes student feedback to determine if the sentiment is **positive**, **negative**, or **neutral**. It uses a sophisticated ensemble model combining DeBERTa, SVM, and LightGBM.")

st.subheader("Enter Feedback Text")
user_input = st.text_area("Please enter the feedback you want to analyze:", height=150, placeholder="e.g., 'The professor was engaging and the course materials were very helpful.'")

if st.button("Analyze Sentiment"):
    if predictor is None:
        st.stop()

    # --- Special Rule: Check for input length ---
    if len(user_input.split()) < 3:
        st.warning("⚠️ **Out of Context:** Input is too short. Please enter at least 3 words for a meaningful analysis.")
    else:
        with st.spinner("Analyzing..."):
            # --- Perform Prediction ---
            predicted_label, prob_dict = predictor.predict_proba(user_input)

            # --- Display Results ---
            st.subheader("Analysis Results")
            
            # Display final sentiment with an icon
            if predicted_label == 'positive':
                st.success(f"**Predicted Sentiment: Positive** 👍")
            elif predicted_label == 'negative':
                st.error(f"**Predicted Sentiment: Negative** 👎")
            else:
                st.info(f"**Predicted Sentiment: Neutral** 😐")

            # Display prediction confidence percentages
            st.markdown("#### Prediction Confidence")
            for label, prob in prob_dict.items():
                st.write(f"**{label.capitalize()}**")
                st.progress(prob, text=f"{prob:.2%}")

            # Display word importance visualization
            st.markdown("#### Key Word Influences")
            st.markdown(
                """
                <small>The highlight intensity suggests words that were influential to the model's TF-IDF component. 
                This is an approximation of the model's focus.</small>
                """, unsafe_allow_html=True
            )
            highlighted_html = highlight_word_importance(user_input, predictor, predicted_label)
            st.markdown(f"<div style='border: 1px solid #ddd; border-radius: 5px; padding: 10px; line-height: 2.0;'>{highlighted_html}</div>", unsafe_allow_html=True)