import torch

import torch.nn.functional as F

# From https://github.com/karpathy/nanoGPT/blob/master/train.py
def get_batch(data, sequence_length, batch_size, device='cpu'):
    device_type = device
    ix = torch.randint(len(data) - sequence_length, (batch_size,))
    x = torch.stack([(data[i:i+sequence_length]) for i in ix])
    y = torch.stack([(data[i+1:i+1+sequence_length]) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Training function.
def train_step(
    model, 
    dataset_train, 
    optimizer, 
    criterion, 
    sequence_length, 
    vocab_size,
    batch_size,
    device
):
    model.train()
    inputs, labels = get_batch(
        dataset_train, sequence_length, batch_size, device
    )
    optimizer.zero_grad()
    # Forward pass.
    outputs = model(inputs)
    
    labels = labels.contiguous().view(-1)
    outputs = outputs.view(-1, vocab_size)
    # Calculate the loss.
    loss = criterion(
        outputs, 
        labels.type(torch.int64)
    )
    # Backpropagation.
    loss.backward()
    # Update the optimizer parameters.
    optimizer.step()
    return loss

# Validation function.
def val_step(
    model, 
    dataset_valid, 
    criterion, 
    sequence_length, 
    vocab_size,
    batch_size, 
    device
):
    model.eval()
    inputs, labels = get_batch(
        dataset_valid, sequence_length, batch_size, device
    )
    # Forward pass.
    with torch.no_grad():
        outputs = model(inputs)

    labels = labels.contiguous().view(-1)
    outputs = outputs.view(-1, vocab_size)
    # Calculate the loss.
    loss = criterion(
        outputs, 
        labels.type(torch.int64)
    )
    return loss

class NLPDataset():
    def __init__(self, file_path, enc):
        self.file_path = file_path
        self.text_file = open(file_path)
        self.lines = self.text_file.read()
        self.enc = enc
    
    def __len__(self):
        return len(self.file_paths)

    def get_data(self):
        final_vector = self.enc.encode(self.lines)
        return torch.tensor(final_vector[0::], dtype=torch.int32)