import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Real news samples (50+ samples for better diversity)
real_news = [
    'Government announces new climate policies for 2050',
    'Scientists discover new species in rainforest',
    'Stock market rises on positive economic data',
    'Vaccine shows promising clinical trial results',
    'Trade agreement signed between countries',
    'Health officials announce disease milestone',
    'University achieves quantum computing breakthrough',
    'Infrastructure bill approved by parliament',
    'Federal Reserve maintains current interest rates',
    'Tech company launches sustainability initiative',
    'WHO recommends new health screening protocols',
    'Researchers publish renewable energy findings',
    'Economy shows signs of recovery',
    'New employment regulations passed',
    'Summit addresses international climate change',
    'Pharma company develops new treatment',
    'Space agency launches successful mission',
    'Medical research finds cancer therapy',
    'Government invests in education',
    'Markets see steady growth',
    'New study shows health benefits of exercise',
    'City council approves budget for public transit',
    'University grants increase research funding',
    'Company announces record quarterly earnings',
    'Agriculture department promotes sustainable farming',
    'Tech startup receives venture capital funding',
    'Hospital implements new patient safety protocol',
    'Schools receive new technology equipment',
    'Environmental agency reports on air quality',
    'Transportation department upgrades infrastructure',
    'Bank announces new financial services',
    'Researchers find new renewable energy source',
    'Government launches job training program',
    'Utility company expands electric vehicle charging',
    'Medical breakthrough in disease treatment',
    'Sports team wins championship',
    'Community center opens new facility',
    'Airline announces new route',
    'Manufacturing plant creates new jobs',
    'Scientific expedition discovers new findings',
    'Network of cities commit to environmental goals',
    'University completes major renovation project',
    'Police department implements reform measures',
    'News organization investigates corporate practices',
    'Education initiative improves student literacy',
    'Travel industry recovers from pandemic',
    'Technology company expands workforce',
    'City celebrates record tourism numbers',
    'Research shows benefits of new therapy',
    'Government agency implements new regulations',
]

# Fake news samples (50+ samples for diversity)
fake_news = [
    'Aliens land in major city, government denies',
    'Celebrities secretly control governments',
    'Miracle cure hidden by pharmaceutical companies',
    'Moon landing was fake conspiracy',
    'One food will change your life forever',
    'Secret billionaires control everything',
    'Simple trick makes you rich overnight',
    'Government weapons turning frogs gay',
    'Fountain of youth discovered on island',
    'New world order plans exposed',
    'Celebrity death was hoax',
    'Doctors hate this weight loss trick',
    'Government implants microchips in vaccines',
    'Secret society runs world',
    'Flat earth truth finally revealed',
    'NASA faking all space missions',
    'Actor admits satanic rituals',
    'Royal family has alien technology',
    'Ancient aliens built all structures',
    'Underground bunkers for billionaires',
    'This supplement cures all diseases',
    'Illuminati controls major events',
    'Famous person secretly survived',
    'Government tracking everyone 24/7',
    'Shocking evidence proves conspiracy',
    'Doctors dont want you knowing this',
    'Elites hiding immortality serum',
    'This one ingredient will change everything',
    'Mass hypnosis by media confirmed',
    'Reptilians exposed as world leaders',
    'Underground city discovered',
    'Time travelers among us confirmed',
    'Ancient prophecy predicted this',
    'Psychic powers scientifically proven',
    'Weather control technology exists',
    'Celebrities part of secret cult',
    'Government uses mind control',
    'Ancient bloodline rules everything',
    'Miracle diet doctors forbid',
    'Banking collapse imminent',
    'Shocking truth about history',
    'Cloning technology perfected',
    'Five major world events explained',
    'Celebrity admits the truth',
    'Hidden technology will change world',
    'Shocking video proves theory',
    'Scientists find evidence of cover up',
    'This changes everything you know',
    'Viral claim about famous person',
    'Major incident completely fabricated',
    'Insider reveals shocking secrets',
]

# Create balanced dataset - 50 real + 50 fake = 100 total samples
texts = real_news + fake_news
labels = [1] * len(real_news) + [0] * len(fake_news)

df = pd.DataFrame({'text': texts, 'label': labels})

# Shuffle data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset: {len(df)} samples")
print(f"Real news: {sum(df['label'] == 1)}")
print(f"Fake news: {sum(df['label'] == 0)}")

# Split data - 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=1,
    max_df=1.0,
    lowercase=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Features created: {X_train_vec.shape[1]}")

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\n=== Model Performance ===")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print(f"\nModel and vectorizer saved successfully!")
