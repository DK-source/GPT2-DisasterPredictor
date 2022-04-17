import os
import joblib
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer

class TweetsDataset(Dataset):
    def __init__(self,path):
        super().__init__()
        self.data = pd.read_csv(path)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        record = self.data.iloc[index]
        text = record['text']
        return {'text': text, 'label': '0'}

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

model_name = 'gpt2'
max_length = 60
batch_size = 32
model = joblib.load('./Training/GPT2model.pkl')
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
gpt2_classificaiton_collator = Gpt2ClassificationCollator(tokenizer,max_length)


path = input('Enter the path to the dataset that you want to predict: ')
device = 'cpu'

dataset = TweetsDataset(path)
dataloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=gpt2_classificaiton_collator)

test_labels, test_predict, test_loss = validation(dataloader, device)

original = pd.read_csv(path)
original.to_csv('./Output/Predicted_dataset.csv')
predicted = pd.read_csv('./Output/Predicted_dataset.csv')
predicted['target'] = test_predict
predicted.to_csv('./Output/Predicted_dataset.csv', index=False)
print('''Prediction has finished please check the 'Output' folder for your predicted dataset.''')