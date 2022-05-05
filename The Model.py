# Basic
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
# GPT-2
import torch
import torch.nn.functional as F
import tensorflow as tf
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (set_seed,
                          GPT2Config,
                          GPT2ForSequenceClassification,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup)

# Basic setting
epochs = 10
batch_size = 32
max_length = 60
device = 'cpu'
model_name = 'gpt2'

# Model
set_seed(654) # For reproductibility
print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

print('Loading model...')
model_config = GPT2Config.from_pretrained(model_name, num_labels=2)
model = GPT2ForSequenceClassification.from_pretrained(model_name, config=model_config)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
model.to(device)

print('Model loaded to `%s`'%device)

# Dataset
class TweetsDataset(Dataset):
    def __init__(self, train=True):
        self.train = train
        self.data = pd.read_csv(os.path.join('./Datasets', 'train.csv' if train else 'test.csv'))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        record = self.data.iloc[index]
        text = record['text']
        if self.train:
            return {'text': text, 'label': record['target']}
        else:
            return {'text': text, 'label': '0'}

# Collator
class Gpt2ClassificationCollator(object):
    def __init__(self, tokenizer, labels_encoder, max_seq_len=None):
        self.tokenizer = tokenizer # Tokenizer to be used inside the class.
        self.max_seq_len =tokenizer.model_max_length if max_seq_len is None else max_seq_len # Check max sequence length.
    def __call__(self, sequences):
        data = [sequence['text'] for sequence in sequences] # Get all texts from sequences list.
        labels = [int(sequence['label']) for sequence in sequences] # Get all labels from sequences list.
        # Call tokenizer on all texts to convert into tensors of numbers with appropriate padding.
        inputs = self.tokenizer(text=data,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=self.max_seq_len)
        inputs.update({'labels':torch.tensor(labels)}) # Update the inputs with the associated encoded labels as tensor.
        return inputs

# Train and Validation function
def train(dataloader, optimizer, scheduler, device_):
    global model
    model.train()
    prediction_labels = []
    true_labels = []
    total_loss = []
    for batch in dataloader:
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device_) for k, v in batch.items()}
        outputs = model(**batch)
        loss, logits = outputs[:2]
        logits = logits.detach().cpu().numpy()
        total_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        prediction_labels += logits.argmax(axis=-1).flatten().tolist()
    return true_labels, prediction_labels, total_loss

def validation(dataloader, device_):
    global model
    model.eval()
    prediction_labels = []
    true_labels = []
    total_loss = []
    for batch in dataloader:
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device_) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            total_loss.append(loss.item())
            prediction_labels += logits.argmax(axis=-1).flatten().tolist()
    return true_labels, prediction_labels, total_loss

# Instatiate Collator
gpt2_classificaiton_collator = Gpt2ClassificationCollator(tokenizer,max_length)

# Dataloader for datasets
print('Dealing with Train...')
train_dataset = TweetsDataset(train=True)
print('Created `train_dataset` with %d examples!'%len(train_dataset))
train_size = int(len(train_dataset) * 0.8)
val_size = len(train_dataset) - train_size
train_dataset, valid_dataset = random_split(train_dataset, [train_size, val_size])
print('Created `valid_dataset` with %d examples!'%len(valid_dataset))
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=gpt2_classificaiton_collator)
print('Created `train_dataloader` with %d batches!'%len(train_dataloader))
valid_dataloader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=gpt2_classificaiton_collator)
print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))

#Optimizer and Scheduler
optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                  )
num_train_steps = len(train_dataloader) * epochs
num_warmup_steps = int(num_train_steps * 0.1)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = num_warmup_steps, # Default value in run_glue.py
                                            num_training_steps = num_train_steps)

# Training
for epoch in range(epochs):
    print('Training on batches...')
    # Perform one full pass over the training set.
    train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device)
    train_acc = accuracy_score(train_labels, train_predict)

    # Get prediction form model on validation data.
    print('Validation on batches...')
    valid_labels, valid_predict, valid_loss = validation(valid_dataloader, device)
    valid_acc = accuracy_score(valid_labels, valid_predict)
    print(f'Epoch: {epoch}, train_loss: {torch.tensor(train_loss).mean():.3f}, train_acc: {train_acc:.3f}, val_loss: {torch.tensor(valid_loss).mean():.3f}, val_acc: {valid_acc:.3f}')

# Exporting Model
joblib.dump(model, 'GPT2Model.pkl')