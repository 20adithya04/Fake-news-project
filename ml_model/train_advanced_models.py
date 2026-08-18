"""Train one of the advanced fake-news classifiers.

Expected CSV columns: ``text`` and ``label`` (label: 0 = fake, 1 = real).
Example: python train_advanced_models.py --algorithm lstm --data data/news.csv

The script intentionally trains one model per run.  Training every deep-learning
model at once is slow and makes it harder to compare experiments fairly.
"""
import argparse
import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "trained_models"


def load_demo_data():
    """Read the existing demo samples without executing train_model.py."""
    source = (ROOT / "train_model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    samples = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {"real_news", "fake_news"}):
            samples[node.targets[0].id] = ast.literal_eval(node.value)
    if set(samples) != {"real_news", "fake_news"}:
        raise ValueError("Could not load the built-in demo data.")
    return pd.DataFrame({
        "text": samples["real_news"] + samples["fake_news"],
        "label": [1] * len(samples["real_news"]) + [0] * len(samples["fake_news"]),
    })


def load_data(path):
    if path is None:
        print("Using the built-in demo dataset. Use --data path/to/news.csv for meaningful results.")
        frame = load_demo_data()
    else:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Dataset not found: {path}. Provide a valid CSV path, or omit --data to use the demo dataset."
            )
        frame = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError("Dataset must contain text and label columns (0=fake, 1=real).")
    frame = frame[["text", "label"]].dropna()
    frame["text"] = frame["text"].astype(str)
    frame["label"] = frame["label"].astype(int)
    if len(frame) < 100:
        print("Warning: fewer than 100 samples. Deep-learning/transformer results will not be reliable.")
    return train_test_split(frame["text"].to_numpy(), frame["label"].to_numpy(),
                            test_size=0.2, random_state=42, stratify=frame["label"])


def scores(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def train_lightgbm(x_train, y_train, x_test):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError("Install LightGBM: pip install lightgbm") from exc
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
    train_vectors = vectorizer.fit_transform(x_train)
    test_vectors = vectorizer.transform(x_test)
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           random_state=42, class_weight="balanced", verbosity=-1)
    model.fit(train_vectors, y_train)
    return model.predict(test_vectors), {"model": model, "vectorizer": vectorizer}


def train_neural(kind, x_train, y_train, x_test):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Install TensorFlow: pip install tensorflow") from exc
    tf.keras.utils.set_random_seed(42)
    vectorize = tf.keras.layers.TextVectorization(max_tokens=20000, output_mode="int",
                                                   output_sequence_length=300)
    vectorize.adapt(x_train)
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    x = vectorize(inputs)
    x = tf.keras.layers.Embedding(20000, 128, mask_zero=(kind != "cnn"))(x)
    if kind == "gru":
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64))(x)
    elif kind == "lstm":
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    else:  # cnn
        x = tf.keras.layers.Conv1D(128, 5, activation="relu")(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                                   restore_best_weights=True)]
    model.fit(np.asarray(x_train).reshape(-1, 1), y_train, validation_split=0.15,
              epochs=10, batch_size=32, callbacks=callbacks, verbose=1)
    probabilities = model.predict(np.asarray(x_test).reshape(-1, 1), verbose=0).ravel()
    return (probabilities >= 0.5).astype(int), model


def train_transformer(kind, x_train, y_train, x_test, y_test, output_dir):
    try:
        from datasets import Dataset
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  DataCollatorWithPadding, Trainer, TrainingArguments)
    except ImportError as exc:
        raise RuntimeError("Install transformer dependencies: pip install transformers datasets torch") from exc
    checkpoint = "distilbert-base-uncased" if kind == "bert" else "FacebookAI/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    def tokenise(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)
    train_set = Dataset.from_dict({"text": list(x_train), "label": list(y_train)}).map(tokenise, batched=True)
    test_set = Dataset.from_dict({"text": list(x_test), "label": list(y_test)}).map(tokenise, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    args = TrainingArguments(output_dir=str(output_dir / "checkpoints"), num_train_epochs=3,
                             per_device_train_batch_size=8, per_device_eval_batch_size=8,
                             learning_rate=2e-5, report_to="none", save_strategy="no")
    trainer = Trainer(model=model, args=args, train_dataset=train_set,
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer))
    trainer.train()
    logits = trainer.predict(test_set).predictions
    predictions = np.argmax(logits, axis=1)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True,
                        choices=["lightgbm", "gru", "lstm", "cnn", "bert", "roberta"])
    parser.add_argument("--data", help="CSV with text and label columns; omit to use built-in demo data")
    args = parser.parse_args()
    x_train, x_test, y_train, y_test = load_data(args.data)
    MODELS_DIR.mkdir(exist_ok=True)
    output_dir = MODELS_DIR / args.algorithm

    if args.algorithm == "lightgbm":
        predictions, artifact = train_lightgbm(x_train, y_train, x_test)
        output_dir.mkdir(exist_ok=True)
        joblib.dump(artifact, output_dir / "model.pkl")
        model_format = "joblib_tfidf"
    elif args.algorithm in {"gru", "lstm", "cnn"}:
        predictions, model = train_neural(args.algorithm, x_train, y_train, x_test)
        output_dir.mkdir(exist_ok=True)
        model.save(output_dir / "model.keras")
        model_format = "keras"
    else:
        predictions = train_transformer(args.algorithm, x_train, y_train, x_test, y_test, output_dir)
        model_format = "transformers"

    report = {"algorithm": args.algorithm, "format": model_format, "metrics": scores(y_test, predictions)}
    (output_dir / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {args.algorithm} model to {output_dir}")


if __name__ == "__main__":
    main()
