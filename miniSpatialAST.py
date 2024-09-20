import os
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Config 
sample_rate = 48000
n_mels = 64
n_fft = 1024
hop_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Direction labels
direction_mapping = {
    "front": 0,
    "back": 1,
    "left": 2,
    "right": 3
}

 # Dataset class
class SoundDataset(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sound_path = self.data[idx]
        label = self.labels[idx]

        # Load stereo audio
        waveform, sr = torchaudio.load(sound_path)

        # Ensure the audio is in stereo and resample if necessary
        if waveform.shape[0] != 2:
            raise ValueError(f"Expected stereo sound, got {waveform.shape[0]} channels")
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)

        # Convert to Mel-spectrogram
        mel_spectrogram_left = librosa.feature.melspectrogram(y=waveform[0].numpy(), sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        mel_spectrogram_right = librosa.feature.melspectrogram(y=waveform[1].numpy(), sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        
        # Combine left and right mel-spectrograms (stack them along a new axis)
        mel_spectrogram = np.stack([mel_spectrogram_left, mel_spectrogram_right], axis=0)

        # Compute IPD (Interaural Phase Difference)
        stft_left = librosa.stft(waveform[0].numpy(), n_fft=n_fft, hop_length=hop_length)
        stft_right = librosa.stft(waveform[1].numpy(), n_fft=n_fft, hop_length=hop_length)

        # Calculate the phase difference (IPD)
        phase_left = np.angle(stft_left)
        phase_right = np.angle(stft_right)
        ipd = np.angle(np.exp(1j * (phase_right - phase_left)))

        # Ensure IPD and Mel-spectrogram have the same shape
        ipd_resized = librosa.util.fix_length(ipd, mel_spectrogram.shape[-1], axis=-1)

        # Stack the Mel-spectrogram and IPD along the channel dimension
        features = np.stack([mel_spectrogram, ipd_resized], axis=0)

        return torch.tensor(features, dtype=torch.float32), label
    

# Model architecture
class MiniSpatialAST(nn.Module):
    def __init__(self, n_classes=4):
        super(MiniSpatialAST, self).__init__()

        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * (n_mels // 8) * 32, 128)  # Adjust based on input size
        self.fc2 = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load data
def load_data(data_folder):
    data = []
    labels = []
    for direction, label in direction_mapping.items():
        direction_folder = os.path.join(data_folder, direction)
        for file in os.listdir(direction_folder):
            if file.endswith(".wav"):
                data.append(os.path.join(direction_folder, file))
                labels.append(label)
    return data, labels

# Main training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs=10):
    model = model.to(device)
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        val_loss = 0.0
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Accuracy: {accuracy}%")

# Load and split the dataset
data_folder = "./recordings"
data, labels = load_data(data_folder)
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create Datasets and Dataloaders
train_dataset = SoundDataset(train_data, train_labels, transform=transforms.ToTensor())
val_dataset = SoundDataset(val_data, val_labels, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model, criterion, and optimizer
model = MiniSpatialAST(n_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs=10)