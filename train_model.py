# train_model.py
# Run this script to train the Veritas 13.0 Neural Brain (10 Classes)
# Ideally run on a machine with a GPU (e.g., Google Colab)

import torch
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import shutil
import os

print("🧠 Generating 'Top 10' AI Fingerprint Dataset...")

# --- 2. CREATE SYNTHETIC DATA FOR 10 MODELS ---
data_sources = {
    0: ("Human", [
        "I literally can't believe this happened.", "The quick brown fox jumps.",
        "Breaking news from the capital today.", "Hey, can you send that file?",
        "My opinion is that pizza is the best food.", "Idk what to do honestly."
    ]),
    1: ("ChatGPT", [
        "To delve into the intricacies of this topic.", "It is important to note that.",
        "In conclusion, the tapestry of history.", "Furthermore, we must consider.",
        "As an AI language model, I cannot.", "Certainly! Here is the information."
    ]),
    2: ("Gemini", [
        "Here is a comprehensive breakdown:", "* Point 1\n* Point 2",
        "Sure, I can help with that. Key takeaways:", "I have analyzed the request.",
        "**Option 1:** Do this.\n**Option 2:** Do that.", "Here's a summary of the data:"
    ]),
    3: ("Claude", [
        "Certainly.", "Here is the summary you requested.", "In the context of your query.",
        "It is worth noting that.", "Based on the provided text.",
        "I cannot fulfill this request directly, but I can offer..."
    ]),
    4: ("Llama", [
        "Sure! I can help you with that.", "Here's the info you asked for!",
        "Let's dive into it.", "Happy to help!", "Here's a list of things to consider:",
        "It's great that you're interested in this topic!"
    ]),
    5: ("Mistral", [
        "Here is the code.", "The answer is 42.", "Do this:",
        "1. Step one\n2. Step two", "Consise explanation:", "No unnecessary fluff."
    ]),
    6: ("Grok", [
        "Here's the deal.", "Remember, this is just a simulation.",
        "It might be X, but also consider Y.", "Let's look at the facts.",
        "Funny thing about that is..."
    ]),
    7: ("DeepSeek", [
        "The technical implementation requires...", "```python\ndef code():\n```",
        "Analyzed logic:", "Mathematical proof:", "Optimization suggestion:"
    ]),
    8: ("Copilot", [
        "I found some results for you.", "According to the web search.",
        "Here is a creative poem about that.", "I can generate that image for you."
    ]),
    9: ("Perplexity AI", [                       # <--- UPDATED TRAINING DATA
        "According to Sources 1 and 2.", "Found 5 results for your query.",
        "The search results indicate.", "Based on the web page context.",
        "[1] Citation needed.", "Here is the answer based on recent search:"
    ])
}

# Generate 500 examples for EACH of the 10 models (5,000 total training samples)
combined_text = []
combined_labels = []

for label_id, (name, phrases) in data_sources.items():
    # Repeat the phrases to create volume
    current_texts = phrases * 85  
    combined_text.extend(current_texts)
    combined_labels.extend([label_id] * len(current_texts))

dataset = Dataset.from_dict({"text": combined_text, "label": combined_labels})
dataset = dataset.train_test_split(test_size=0.1)

print(f"✅ Generated {len(dataset['train'])} training examples across 10 categories.")

# --- 3. TRAIN THE MULTI-CLASS MODEL ---
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=10) 

training_args = TrainingArguments(
    output_dir="./veritas_model_10x",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    num_train_epochs=3, 
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

print("🚀 Starting 10-Model Training...")
trainer.train()

# --- 4. SAVE ---
print("💾 Saving Veritas 13.0 Brain...")
model.save_pretrained("./veritas_model")
tokenizer.save_pretrained("./veritas_model")
