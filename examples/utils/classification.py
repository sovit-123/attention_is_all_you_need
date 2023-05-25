import torch
import string
import numpy as np
import os
import pathlib
import re

from tqdm.auto import tqdm
from torch.utils.data import Dataset

class NLPClassificationDatasetCustom(Dataset):
    def __init__(self, file_paths, word_frequency, int_mapping, max_len):
        self.word_frequency = word_frequency
        self.int_mapping = int_mapping
        self.file_paths = file_paths
        self.max_len = max_len

    def standardize_text(self, input_text):
        # Convert everything to lower case.
        text = input_text.lower()
        # If the text contains HTML tags, remove them.
        text = re.sub('<[^>]+>+', '', text)
        # Remove punctuation marks using `string` module.
        # According to `string`, the following will be removed,
        # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        text = ''.join([
            character for character in text \
                if character not in string.punctuation
        ])
        return text

    def return_int_vector(self, int_mapping, text_file_path):
        """
        Assign an integer to each word and return the integers in a list.
        """
        with open(text_file_path, 'r') as f:
            text = f.read()
            text = self.standardize_text(text)
            corpus = [
                word for word in text.split()
            ] 
        # Each word is replaced by a specific integer.
        int_vector = [
            int_mapping[word] for word in text.split() \
            if word in int_mapping
        ]
        return int_vector
    
    def pad_features(self, int_vector, max_len):
        """
        Return features of `int_vector`, where each vector is padded 
        with 0's or truncated to the input seq_length. Return as Numpy 
        array.
        """
        features = np.zeros((1, max_len), dtype = int)
        if len(int_vector) <= max_len:
            zeros = list(np.zeros(max_len - len(int_vector)))
            new = zeros + int_vector
        else:
            new = int_vector[: max_len]
        features = np.array(new)
        return features

    def encode_labels(self, text_file_path):
        file_path = pathlib.Path(text_file_path)
        class_label = str(file_path).split(os.path.sep)[-2]
        if class_label == 'pos':
            int_label = 1
        else:
            int_label = 0
        return int_label

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        int_vector = self.return_int_vector(self.int_mapping, file_path)
        padded_features = self.pad_features(int_vector, self.max_len)
        label = self.encode_labels(file_path)
        return {
            'text': torch.tensor(padded_features, dtype=torch.int32),
            'label': torch.tensor(label, dtype=torch.long)
        }

def binary_accuracy(labels, outputs, train_running_correct):
    # As the outputs are currently logits.
    outputs = torch.sigmoid(outputs)
    running_correct = 0
    for i, label in enumerate(labels):
        if label < 0.5 and outputs[i] < 0.5:
            running_correct += 1
        elif label >= 0.5 and outputs[i] >= 0.5:
            running_correct += 1
    return running_correct

# Training function.
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        inputs, labels = data['text'], data['label']
        inputs = inputs.to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(inputs)
        outputs = torch.squeeze(outputs, -1)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        running_correct = binary_accuracy(
            labels, outputs, train_running_correct
        )
        train_running_correct += running_correct
        # Backpropagation.
        loss.backward()
        # Update the optimizer parameters.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# Validation function.
def validate(model, testloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            inputs, labels = data['text'], data['label']
            inputs = inputs.to(device)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            # Forward pass.
            outputs = model(inputs)
            outputs = torch.squeeze(outputs, -1)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            running_correct = binary_accuracy(
                labels, outputs, valid_running_correct
            )
            valid_running_correct += running_correct
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc