import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix


# CONFIG

MODEL_DIR = "./camembert_valdataset"

TEXT_COL = "clean_comment"


# LOAD MODEL + THRESHOLDS

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

THRESHOLDS = model.config.thresholds
LABEL_COLS = model.config.label_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# UTILS

def tokenize(batch):
    return tokenizer(
        batch[TEXT_COL],
        truncation=True,
        padding="max_length",
        max_length=256
    )

def pack_labels(batch):
    batch["labels"] = np.stack(
        [batch[col] for col in LABEL_COLS],
        axis=1
    ).astype(np.float32)
    return batch

def prepare(df):
    ds = Dataset.from_pandas(df[[TEXT_COL] + LABEL_COLS])
    ds = ds.map(tokenize, batched=True)
    ds = ds.map(pack_labels, batched=True)
    ds = ds.remove_columns(LABEL_COLS + [TEXT_COL])
    ds.set_format("torch")
    return ds

def predict(ds):
    all_labels = []
    all_probs = []

    for batch in ds:
        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        attention_mask = batch["attention_mask"].unsqueeze(0).to(device)
        labels = batch["labels"].unsqueeze(0)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        probs = torch.sigmoid(logits)

        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    preds = np.zeros_like(all_probs, dtype=int)
    for i, t in enumerate(THRESHOLDS):
        preds[:, i] = (all_probs[:, i] > t).astype(int)

    return all_labels, preds


def evaluate(name, df):
    print(f"\n===== {name} =====")

    ds = prepare(df)
    labels, preds = predict(ds)

    print(classification_report(labels, preds, target_names=LABEL_COLS))

    for i, label in enumerate(LABEL_COLS):
        cm = confusion_matrix(labels[:, i], preds[:, i])
        print(f"\nMatrice de confusion — {label}")
        print(cm)


# LOAD DATASETS

df_main = pd.read_csv('./../../data/labelled_topics/dataset_avis.csv')
df_gold = pd.read_csv('./../../data/test_dataset/100_avis_annote.csv', sep=";")


# SPLIT TEST LIKE TRAIN

from sklearn.model_selection import train_test_split

_, df_test = train_test_split(df_main, test_size=0.2, random_state=42)


# EVALUATION

evaluate("TEST SPLIT", df_test)
evaluate("GOLD DATASET", df_gold)