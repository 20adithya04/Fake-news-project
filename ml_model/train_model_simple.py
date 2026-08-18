import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Real news samples
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
]

# Fake news samples  
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
]

# Create balanced dataset
texts = real_news + fake_news
labels = [1] * len(real_news) + [0] * len(fake_news)

df = pd.DataFrame({'text': texts, 'label': labels})

print(f"Dataset: {len(df)} samples")
print(f"Real news: {sum(df['label'] == 1)}")
print(f"Fake news: {sum(df['label'] == 0)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Vectorize
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=1,
    max_df=1.0
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\nModel Performance:")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Save
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("\nModel saved successfully!")
