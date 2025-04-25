# Fake News Detection on Twitter

## Overview

This project focuses on detecting fake news and disinformation on the social media platform **Twitter** using machine learning techniques. The main goal is to explore the predictive power of different types of features—**user-based**, **textual**, and **emotional**—as well as their combinations, in classifying tweets as credible or not.

## Dataset

We used the **TruthSeeker dataset**, provided by the [University of New Brunswick – Canadian Institute for Cybersecurity (UNB CIC)](https://www.unb.ca/cic/datasets/truthseeker-2023.html). The dataset contains labeled tweets along with metadata about the users and tweet content, designed specifically for rumor and misinformation detection.

## Preprocessing

The preprocessing phase included:

- **Text cleaning** (removal of URLs, mentions, hashtags, punctuation, and lowercasing)
- **Stop word removal**
- **Tokenization**
- **Log-scaling** of certain numerical features to reduce skewness
- **Feature engineering** using the **Empath** library to extract emotional and semantic categories from tweet content

## Feature Groups

The classification task was carried out over three distinct groups of features:

1. **User-based attributes** – e.g., follower count, credibility score, total likes
2. **Textual features** – e.g., average word count, min word length, presence of 
3. **Emotional categories** - derived using the **Empath** library and grouped (with the help of ChatGPT) into five broader groups: emotion_positive, emotion_negative, emotion_social, emotion_intense and emotion_cognitive.

Additionally, a combined dataset including all features was used to evaluate the performance when all available information is considered.

## Text Representation + Classification

For the textual feature group, tweet content was also transformed into vector form using **TF-IDF**, **Word2Vec** and **FastText** embeddings. This allowed us to test how raw text representation contributes to classification performance.

## Classification Models

Seven different machine learning algorithms were evaluated:

- Logistic Regression
- Naive Bayes
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors (k-NN)

Each model was trained and tested separately on each feature group, as well as on the combined dataset, to observe performance differences and feature importance across algorithms.

## Results & Evaluation

Evaluation metrics such as **accuracy**, **precision**, **recall**, and **F1-score** were used to assess model performance. Feature importance was analyzed to identify which attributes most influenced the classification.

## Citation

Inspired by:

[1] M. Al-Tarawneh, O. Al-irr, K. Al-Maaitah, H. Kanj and W. Aly, „Enhancing Fake News Detection with Word Embedding: A Machine Learning and Deep Learning Approach,“ Preprints, 2024. https://doi.org/10.20944/preprints202407.2317.v1

[2] X. Zhou, A. Jain, V. Phoha and R. Zafarani, “Fake News Early Detection: A Theory-driven Model”, 2019. https://arxiv.org/abs/1904.11679
