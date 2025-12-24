import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Updated Dummy Dataset (Real/Fake/Neutral)
# 0=Real, 1=Fake, 2=Neutral
DUMMY_DATA = [
    ("The government passed a new law.", 0),        # Real
    ("Aliens are eating the moon.", 1),             # Fake
    ("Hi, how are you doing today?", 2),            # Neutral
    ("I am just testing this app.", 2),             # Neutral
    ("My name is Saurabh.", 2),                     # Neutral
    ("The stock market crashed.", 0),               # Real
    ("Drinking bleach is healthy.", 1)              # Fake
]

MODEL_NAME = "distilbert-base-uncased"
MODEL_PATH = "fact_check_model_v2" 

class FactDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_fact_checker():
    print("Starting training with dummy dataset (3 classes)...")
    texts = [item[0] for item in DUMMY_DATA]
    labels = [item[1] for item in DUMMY_DATA]

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = FactDataset(train_encodings, train_labels)
    val_dataset = FactDataset(val_encodings, val_labels)

    # num_labels=3 for Real, Fake, Neutral
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    training_args = TrainingArguments(
        output_dir='./results_v2',
        num_train_epochs=15,             # Increased epochs for better convergence on small data
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs_v2',
        logging_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    
    print(f"Saving model to {MODEL_PATH}...")
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print("Training complete and model saved.")

def predict_factuality(text):
    if not os.path.exists(MODEL_PATH):
        return "Model not found. Please train first."

    try:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        predicted_class_id = logits.argmax().item()
        
        # Mapping: 0=Real, 1=Fake, 2=Neutral
        if predicted_class_id == 0:
            return "REAL"
        elif predicted_class_id == 1:
            return "FAKE"
        else:
            return "NEUTRAL"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    train_fact_checker()
