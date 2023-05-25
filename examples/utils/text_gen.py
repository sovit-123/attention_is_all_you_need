import torch

from tqdm.auto import tqdm

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
def train(
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
    print('Training')
    train_running_loss = 0.0
    counter = 0
    for i in tqdm(
        range(0, dataset_train.size(0), sequence_length), 
        total=int(dataset_train.size(0)/sequence_length)
    ):
        counter += 1
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
        train_running_loss += loss.item()
        # Backpropagation.
        loss.backward()
        # Update the optimizer parameters.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    return epoch_loss

# Validation function.
def validate(
    model, 
    dataset_valid, 
    criterion, 
    sequence_length, 
    vocab_size,
    batch_size, 
    device
):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    counter = 0
    
    for i in tqdm(
        range(0, dataset_valid.size(0), sequence_length), 
        total=int(dataset_valid.size(0)/sequence_length)
    ):
        counter += 1
        inputs, labels = get_batch(
            dataset_valid, sequence_length, batch_size, device
        )
        # Forward pass.
        with torch.no_grad():
            outputs = model(inputs)
    
        labels = labels.contiguous().view(-1)
        # Calculate the loss.
        loss = criterion(
            outputs.view(-1, vocab_size), 
            labels.type(torch.int64)
        )
        valid_running_loss += loss.item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    return epoch_loss

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