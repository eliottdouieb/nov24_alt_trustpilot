import numpy as np
import pandas as pd
import torch
import json

from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# CONFIG

MODEL_NAME = "camembert-base"
TEXT_COL = "clean_comment"
LABEL_COLS = ["qualité produit", "service livraison", "service client"]

OUTPUT_DIR = "./camembert_valdataset"


# LOAD DATA

df = pd.read_csv('./../../data/labelled_topics/dataset_avis.csv')

dataset = Dataset.from_pandas(df[[TEXT_COL] + LABEL_COLS])

dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = dataset["train"]
test_ds = dataset["test"]

dataset = train_ds.train_test_split(test_size=0.15, seed=42)
train_ds = dataset["train"]
val_ds = dataset["test"]


# TOKENIZATION

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

def prepare(ds):
    ds = ds.map(tokenize, batched=True)
    ds = ds.map(pack_labels, batched=True)
    ds = ds.remove_columns(LABEL_COLS + [TEXT_COL])
    ds.set_format("torch")
    return ds

train_ds = prepare(train_ds)
val_ds = prepare(val_ds)


# MODEL

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_COLS),
    problem_type="multi_label_classification"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    return {
        "f1_micro": f1_score(labels, preds, average="micro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


# TRAINING

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

trainer.train()


# FIND BEST THRESHOLDS

model.eval()
device = next(model.parameters()).device

val_labels = []
val_probs = []

for batch in val_ds:
    input_ids = batch["input_ids"].unsqueeze(0).to(device)
    attention_mask = batch["attention_mask"].unsqueeze(0).to(device)
    labels = batch["labels"].unsqueeze(0)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    probs = torch.sigmoid(logits)

    val_labels.append(labels.numpy())
    val_probs.append(probs.cpu().numpy())

val_labels = np.concatenate(val_labels, axis=0)
val_probs = np.concatenate(val_probs, axis=0)


best_thresholds = []

for i in range(val_labels.shape[1]):
    best_f1 = 0
    best_t = 0.5

    for t in np.arange(0.1, 0.91, 0.01):
        preds = (val_probs[:, i] > t).astype(int)
        f1 = f1_score(val_labels[:, i], preds)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    best_thresholds.append(best_t)

print("Best thresholds:", best_thresholds)


# SAVE MODEL + THRESHOLDS

model.config.thresholds = best_thresholds
model.config.label_names = LABEL_COLS

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)