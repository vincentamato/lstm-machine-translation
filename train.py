import torch
import torch.nn as nn
import torch.optim as optim
from model import create_seq2seq_model, summarize_model
from data import get_data_loader, get_vocab_sizes, TRG, SRC
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
EMBEDDING_DIM = 300
HIDDEN_SIZE = 786
NUM_LAYERS = 3
DROPOUT = 0.4
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
NUM_EPOCHS = 50
CLIP = 0.1
WEIGHT_DECAY = 1e-4

# Train one epoch
def train_epoch(model, iterator, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(iterator, desc="Training", leave=False)
    for src, trg in progress_bar:
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        progress_bar.set_postfix({"batch_loss": f"{loss.item():.3f}"})
    
    return epoch_loss / len(iterator)

# Evaluate one epoch
def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(iterator, desc="Evaluating", leave=False)
        for src, trg in progress_bar:
            src, trg = src.to(device), trg.to(device)
            
            output = model(src, trg, 0)
            
            output = output[:, 1:].reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
            
            progress_bar.set_postfix({"batch_loss": f"{loss.item():.3f}"})
        
    return epoch_loss / len(iterator)

# Get vocabulary sizes
INPUT_SIZE_ENC, INPUT_SIZE_DEC = get_vocab_sizes()
OUTPUT_SIZE = INPUT_SIZE_DEC

# Instantiate the model
model = create_seq2seq_model(SRC, TRG, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, device)
model = model.to(device)
summarize_model(model, INPUT_SIZE_ENC, INPUT_SIZE_DEC)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=TRG['<pad>'])
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Create DataLoaders
train_dataloader = get_data_loader(BATCH_SIZE, split='train')
val_dataloader = get_data_loader(BATCH_SIZE, split='valid')

# Training loop
best_valid_loss = float('inf')
train_losses = []
val_losses = []
patience = 10
epochs_without_improvement = 0

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch: {epoch+1}/{NUM_EPOCHS}")
    
    train_loss = train_epoch(model, train_dataloader, optimizer, criterion, CLIP, device)
    valid_loss = evaluate(model, val_dataloader, criterion, device)
    
    scheduler.step(valid_loss)
    
    train_losses.append(train_loss)
    val_losses.append(valid_loss)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        # Try to save the model, if it fails, use an alternative location
        try:
            torch.save(model.state_dict(), 'best_model.pt')
        except RuntimeError:
            print("Could not save model in the current directory. Trying to save in the user's home directory.")
            home_dir = os.path.expanduser("~")
            save_path = os.path.join(home_dir, 'best_model.pt')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved in {save_path}")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')
    print(f'\tLearning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    
    if epochs_without_improvement == patience:
        print("Early stopping triggered")
        break

# Plot loss curves
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()

plot_losses(train_losses, val_losses)

print("Training completed!")
print("Loss curve saved as 'loss_curve.png'")