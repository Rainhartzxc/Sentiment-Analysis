# Student Feedback Sentiment Analysis using Fine-Tuned DeBERTa

A web-based sentiment analysis application that classifies student feedback into **Positive**, **Neutral**, and **Negative** sentiments using a fine-tuned **DeBERTa Transformer Model**.

This project was developed as part of our undergraduate thesis entitled **"Sentiment Analysis of Student Feedback using Machine Learning and Natural Language Processing."**

---

## Overview

The objective of this project is to automate the analysis of student feedback by utilizing Natural Language Processing (NLP) and Deep Learning techniques. The application allows users to enter textual feedback and instantly predicts the sentiment with a confidence score.

The application was developed using **Python**, **PyTorch**, **Hugging Face Transformers**, and **Streamlit**.

---

## Features

- Fine-tuned Microsoft DeBERTa Transformer model
- Interactive Streamlit web application
- Sentiment prediction:
  - Positive
  - Neutral
  - Negative
- Confidence score for every prediction
- Rule-based validation for insufficient feedback (Less than 3 words)
- Fast inference using cached model loading

---

## Technologies Used

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- DeBERTa
- NumPy
- Joblib

---

## Project Structure

```
Fine-Tuned/
│
├── app.py
├── requirements.txt
├── sentiment_deberta_model/
│
├── notebooks/
│   └── Fine-tuned.ipynb
│
├── data/
│
└── README.md
```

---

## How It Works

1. User enters student feedback.
2. The application validates the input.
3. The DeBERTa tokenizer converts the text into tokens.
4. The fine-tuned DeBERTa model predicts the sentiment.
5. Softmax probabilities are calculated.
6. The predicted label and confidence score are displayed.

---

## Sample Workflow

```
Student Feedback

↓

Tokenizer

↓

Fine-Tuned DeBERTa

↓

Sentiment Prediction

↓

Positive / Neutral / Negative

↓

Confidence Score
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/student-feedback-sentiment-analysis.git
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

## Model

The trained DeBERTa model is **not included** in this repository because it exceeds GitHub's file size limitations.

To run the application, place the trained model inside

```
sentiment_deberta_model/
```

---

## Future Improvements

- Deploy the application to Streamlit Cloud
- Integrate REST APIs for external applications
- Support multilingual sentiment analysis
- Improve model explainability
- Containerize using Docker

---

## Academic Project

This repository contains part of our undergraduate thesis project.

**Role:** Lead Programmer

Responsibilities included:

- Data preprocessing
- Model fine-tuning
- Model evaluation
- Streamlit application development
- Integration of the trained model into the application

---

## License

This repository is shared for educational and portfolio purposes.

Please do not redistribute or reuse the source code without permission from the authors.
