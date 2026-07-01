# Student Feedback Sentiment Analysis using a Stacked Ensemble Model

A web-based sentiment analysis application that classifies student feedback into **Positive**, **Neutral**, and **Negative** sentiments using a **Stacked Ensemble Deep Learning Model** consisting of **CNN**, **LSTM**, and **DeBERTa**, with **Logistic Regression** serving as the meta-classifier.

This project was developed as part of our undergraduate thesis entitled:

> **Sentiment Analysis of Student Feedback using Deep Learning and Natural Language Processing**

---

## 📖 Overview

This project aims to improve the accuracy of sentiment classification by combining multiple deep learning models into a single ensemble architecture.

Instead of relying on a single classifier, the application leverages the strengths of three different models:

- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- Microsoft DeBERTa Transformer

Their predictions are combined using a Logistic Regression meta-classifier to produce the final sentiment prediction.

The application was developed using **Python**, **Streamlit**, **PyTorch**, **TensorFlow/Keras**, and **Scikit-learn**.

---

## ✨ Features

- Stacked Ensemble sentiment classification
- Combines CNN, LSTM, and DeBERTa predictions
- Logistic Regression meta-classifier
- Interactive Streamlit interface
- Predicts:
  - Positive
  - Neutral
  - Negative
- Displays confidence score
- Input validation for insufficient feedback
- Fast model loading using cached resources

---

## 🛠 Technologies Used

- Python
- Streamlit
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- NumPy
- Pandas
- Joblib

---

## 📂 Project Structure

```
Stack-Ensemble/
│
├── app.py
├── requirements.txt
│
│
├── notebooks/
│   └── ENSEMBLED_POWERHOUSE_MODEL.ipynb
│
├── dataset/
│
└── README.md
```

---

## ⚙️ How It Works

1. User enters student feedback.
2. The text is preprocessed.
3. The feedback is evaluated independently by:
   - CNN
   - LSTM
   - DeBERTa
4. Each model generates its own prediction probabilities.
5. These probabilities are combined into a feature vector.
6. Logistic Regression produces the final sentiment prediction.
7. The application displays the predicted sentiment and confidence score.

---

## 🧠 Ensemble Architecture

```
                Student Feedback
                        │
                        ▼
               Text Preprocessing
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
      CNN             LSTM          DeBERTa
        │               │               │
        └───────────────┼───────────────┘
                        ▼
          Prediction Probabilities
                        │
                        ▼
         Logistic Regression Meta Model
                        │
                        ▼
          Final Sentiment Prediction
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/student-feedback-sentiment-analysis-stack-ensemble.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📊 Model

The trained ensemble models are **not included** in this repository because they exceed GitHub's file size limitations.

To run the application, place the trained model files inside:

```
student_feedback_model_ensemble/
```

---

## 📈 Future Improvements

- Deploy to Streamlit Community Cloud
- Support multilingual sentiment analysis
- Add explainable AI (XAI) visualizations
- Optimize inference speed
- Containerize using Docker
- Integrate REST APIs for external applications

---

## 📚 Academic Project

This repository contains part of our undergraduate thesis.

**Role:** Lead Programmer

### Responsibilities

- Designed and implemented the stacked ensemble architecture
- Developed the Streamlit web application
- Integrated CNN, LSTM, and DeBERTa models
- Implemented the Logistic Regression meta-classifier
- Performed data preprocessing and model evaluation
- Conducted testing and application integration

---

## 📷 Application Preview

> Add screenshots of the application inside the `screenshots/` folder.

Example:

```
screenshots/
├── home.png
├── prediction_positive.png
├── prediction_negative.png
└── prediction_neutral.png
```

Then include them in this README.

---

## 📄 License

This repository is shared for educational and portfolio purposes.

Please do not redistribute or reuse the source code without permission from the project authors.

---

## 👨‍💻 Author

**Aldridge Rainhart C. Elumacas**

Lead Programmer

Bachelor of Science in Computer Science  
Major in Intelligent Systems

Laguna State Polytechnic University
