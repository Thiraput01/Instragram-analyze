from torch.utils.data import DataLoader
from dataset import InstagramDataset
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch.nn as nn
from model import Model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
epochs = 20
lr = 0.001

# Load the data
df = pd.read_csv('../data/df_for_nlp.csv')

# normalize the impression count
scaler = StandardScaler()
df['Impressions'] = scaler.fit_transform(df['Impressions'].values.reshape(-1, 1))


# 90% train, 10% test
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# 80% training, 20% validation
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Create datasets for each split
train_dataset = InstagramDataset(train_df)
val_dataset = InstagramDataset(val_df)
test_dataset = InstagramDataset(test_df)

# Create DataLoaders for each split
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for embeddings, labels in train_dataloader:
    print(f"Input to model: {embeddings.shape}")  # Should print (batch_size, embedding_dim)
    print(f"Labels: {labels.shape}")  # Should print the corresponding labels
    break

# Set up the model
model = Model(input_dim=1536)
model = model.to(device)

# Set optimizer and loss function
optimizer = Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss() # For regression tasks
# Modify the train_model function to store losses
def train_model(train_dataloader, val_dataloader, model, epochs=epochs):
    best_val_loss = float('inf')  # Start with an infinitely large validation loss
    best_model_path = "model.pth"
    
    model.train()
    
    train_losses = []  # List to store training losses for plotting
    val_losses = []    # List to store validation losses for plotting
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for embeddings, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(embeddings)
            loss = criterion(outputs.squeeze(), labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Store the training loss for this epoch
        train_losses.append(running_loss / len(train_dataloader))

        # Validation step
        val_loss, val_mse = evaluate_model(val_dataloader, model, criterion, device)
        val_losses.append(val_loss)  # Store the validation loss for this epoch
        cur_loss = running_loss/len(train_dataloader)
        
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {cur_loss}, Validation Loss: {val_loss}, Validation MSE: {val_mse}")
        
        # Save the best model based on validation loss
        if cur_loss < 0.4 and val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Validation loss decreased from {best_val_loss} to {val_loss}. Saving the new best model.")
            torch.save(model.state_dict(), best_model_path)  # Save the model

    print("Training completed.")
    
    print("\nEvaluating on Validation Set...")
    val_loss, val_mse = evaluate_model(val_dataloader, model, criterion, device)
    print(f"Validation Loss: {val_loss}, Validation MSE: {val_mse}")

    print("\nLoading the best model for testing...")
    model.load_state_dict(torch.load(best_model_path))

    print("\nEvaluating on Test Set...")
    test_loss, test_mse, test_mae = test_model(test_dataloader, model, criterion, device)
    print(f"Test Loss: {test_loss}, Test MSE: {test_mse}, Test MAE: {test_mae}")
    print("Test completed.")
    print("-"*50)
    # Plot the loss graph
    plot_loss_graph(train_losses, val_losses)

# Function to plot the training and validation loss
def plot_loss_graph(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


# Define the evaluation function for validation and testing
def evaluate_model(dataloader, model, criterion, device):
    
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_preds = []
    total_labels = []
    
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            total_preds.extend(outputs.squeeze().cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
    
    # Calculate mean squared error or other metrics
    mse = mean_squared_error(total_labels, total_preds)
    return total_loss / len(dataloader), mse

# Define the testing function for evaluating the model on the test set
def test_model(test_dataloader, model, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_preds = []
    total_labels = []
    
    with torch.no_grad():
        for embeddings, labels in test_dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            total_preds.extend(outputs.squeeze().cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
    
    # Calculate Mean Squared Error and Mean Absolute Error
    mse = mean_squared_error(total_labels, total_preds)
    mae = mean_absolute_error(total_labels, total_preds)
    
    print(f"Test Loss: {total_loss / len(test_dataloader)}, Test MSE: {mse}, Test MAE: {mae}")

    return total_loss / len(test_dataloader), mse, mae


# Now call the train_model function
train_model(train_dataloader, val_dataloader, model)