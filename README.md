# 🛒 SuperMart Intelligence Suite

## Overview

SuperMart Intelligence Suite is an AI-powered retail analytics platform that combines multiple Machine Learning and Data Analytics modules to help businesses make data-driven decisions.

The project integrates various intelligent systems including customer segmentation, churn prediction, market basket analysis, sentiment analysis, advertisement optimization, and deep learning models into a single platform.

The objective is to improve customer experience, increase sales, optimize marketing strategies, and provide valuable business insights.

---

# Features

-  Customer Segmentation
-  Market Basket Analysis
-  Sentiment Analysis
-  Customer Churn Prediction
-  Advertisement Optimization
-  Deep Learning Models
-  Interactive Dashboard (Streamlit)

---

# Modules

## 1. Customer Segmentation

Groups customers based on purchasing behavior to help businesses identify target customer groups and personalize marketing campaigns.

**Techniques Used**
- K-Means Clustering
- Data Preprocessing
- Feature Scaling

---

## 2. Market Basket Analysis

Discovers frequently purchased item combinations and generates product recommendations.

**Techniques Used**
- Apriori Algorithm
- Association Rule Mining

---

## 3. Sentiment Analysis

Analyzes customer reviews and classifies them as Positive or Negative.

**Techniques Used**
- Natural Language Processing (NLP)
- NLTK
- CountVectorizer
- Multinomial Naive Bayes

---

## 4. Customer Churn Prediction

Predicts whether a customer is likely to leave the business.

**Techniques Used**
- Machine Learning Classification
- Feature Engineering
- Model Evaluation

---

## 5. Advertisement Optimization

Identifies the most effective advertisements based on customer interactions.

**Techniques Used**
- Reinforcement Learning
- Upper Confidence Bound (UCB)
- Thompson Sampling

---

## 6. Deep Learning

Implements Artificial Neural Networks for predictive analytics tasks.

**Techniques Used**
- TensorFlow
- Keras

---

# Technologies Used

## Programming Language

- Python

## Machine Learning

- Scikit-learn
- TensorFlow
- Keras

## NLP

- NLTK
- CountVectorizer

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn

## Web Application

- Streamlit

## Model Persistence

- Joblib

---

# Project Structure

```
SuperMart/
│
├── app.py
├── data/
│
├── models/
│   ├── sentiment/
│   ├── segmentation/
│   ├── churn_classification/
│   ├── deep_learning/
│   ├── ad_optimization/
│   └── market_basket/
│
├── modules/
│   ├── sentiment.py
│   ├── segmentation.py
│   ├── churn.py
│   ├── market_basket.py
│   ├── advertisement.py
│   └── deep_learning.py
│
├── requirements.txt
└── README.md
```

---

# Workflow

```
               User Input
                    │
                    ▼
           Streamlit Application
                    │
     ┌──────────────┼──────────────┐
     │              │              │
     ▼              ▼              ▼
 Sentiment     Market Basket   Customer Segmentation
     │              │              │
     ▼              ▼              ▼
 Churn Prediction  Advertisement Optimization
                    │
                    ▼
             Deep Learning Models
                    │
                    ▼
            Business Insights
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/Sivaram-Naidu/SuperMart.git
```

Move into the project directory

```bash
cd SuperMart
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

```bash
streamlit run app.py
```

---

# Objectives

- Improve customer satisfaction
- Increase product sales
- Understand customer purchasing patterns
- Reduce customer churn
- Optimize advertisement strategies
- Analyze customer feedback
- Provide intelligent business insights

---

# Future Enhancements

- Real-time analytics
- Cloud deployment
- Advanced transformer-based NLP models
- Recommendation system
- Inventory forecasting
- Sales prediction
- Role-based authentication
- REST API integration

---

# Team Members

- NLP Module
- Market Basket Analysis Module
- Customer Segmentation Module
- Churn Prediction Module
- Advertisement Optimization Module
- Deep Learning Module
- Integration Team

---

# License

This project is developed for academic and educational purposes.

---

# Acknowledgements

- Scikit-learn
- TensorFlow
- Streamlit
- NLTK
- Pandas
- NumPy
