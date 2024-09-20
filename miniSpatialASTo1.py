import os
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1. Prepare the Data

# Define the mapping from subfolder names to labels
angle_bins = ['front', 'back', 'left', 'right']
label_mapping = {'front': 0, 'back': 1, 'left': 2, 'right': 3}
num_angle_bins = len(angle_bins)

# Initialize lists to store file paths and labels
audio_files = []
angle_labels = []

# Path to the 'recordings' folder
recordings_dir = './recordings'

# Loop over each subfolder and collect audio files and labels
for angle_name in angle_bins:
    subfolder = os.path.join(recordings_dir, angle_name)
    if not os.path.isdir(subfolder):
        print(f"Warning: Subfolder {subfolder} does not exist.")
        continue
    # List all files in the subfolder
    for filename in os.listdir(subfolder):
        if filename.lower().endswith(('.wav', '.mp3', '.flac')):  # Include other audio formats if needed
            file_path = os.path.join(subfolder, filename)
            audio_files.append(file_path)
            angle_labels.append(label_mapping[angle_name])

# Ensure that we have audio files to process
if not audio_files:
    raise ValueError("No audio files found in the specified directories.")

print(f"Found {len(audio_files)} audio files.")

# 2. Define the Preprocessing Function

def preprocess_audio(waveform, sample_rate):
    """
    Preprocesses the stereo audio waveform to produce the input tensor Z.

    Args:
        waveform (Tensor): The stereo audio waveform [2, num_samples].
        sample_rate (int): The sample rate of the audio.

    Returns:
        Tensor: The processed input tensor Z [channels, mel_bins, time_steps].
    """
    # Ensure the waveform is stereo
    if waveform.shape[0] != 2:
        raise ValueError("Expected stereo audio with 2 channels.")

    # Parameters
    n_fft = 512
    hop_length = 256
    win_length = 512
    n_mels = 128

    # Compute STFT for both channels
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=None  # To get complex numbers
    )
    X_left = stft_transform(waveform[0])  # [freq_bins, time_steps]
    X_right = stft_transform(waveform[1])

    # Compute magnitude spectrograms
    magnitude_left = X_left.abs()
    magnitude_right = X_right.abs()

    # Compute phase spectrograms
    phase_left = X_left.angle()
    phase_right = X_right.angle()

    # Compute Mel filter bank
    mel_scale = torchaudio.transforms.MelScale(
        n_mels=n_mels,
        sample_rate=sample_rate,
        n_stft=n_fft // 2 + 1
    )

    # Compute Mel-Spectrograms
    mel_left = mel_scale(magnitude_left)  # [mel_bins, time_steps]
    mel_right = mel_scale(magnitude_right)

    # Compute IPD
    ipd = phase_left - phase_right  # [freq_bins, time_steps]
    ipd_cos = torch.cos(ipd)
    ipd_sin = torch.sin(ipd)

    # Apply Mel filter bank to IPD components
    mel_ipd_cos = mel_scale(ipd_cos)
    mel_ipd_sin = mel_scale(ipd_sin)

    # Concatenate the processed components
    Z = torch.cat([
        mel_left.unsqueeze(0),
        mel_right.unsqueeze(0),
        mel_ipd_cos.unsqueeze(0),
        mel_ipd_sin.unsqueeze(0)
    ], dim=0)  # [4, mel_bins, time_steps]

    return Z

# 3. Create the Custom Dataset

class SpatialAudioDataset(Dataset):
    def __init__(self, audio_files, labels, sample_rate=16000):
        """
        Args:
            audio_files (list): List of paths to audio files.
            labels (list): List of angle labels corresponding to the audio files.
            sample_rate (int): The sample rate for loading audio.
        """
        self.audio_files = audio_files
        self.labels = labels
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Load audio
        file_path = self.audio_files[idx]
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            raise IOError(f"Error loading audio file {file_path}: {e}")

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Ensure the waveform is stereo
        if waveform.shape[0] != 2:
            # If mono, duplicate the channel
            waveform = waveform.repeat(2, 1)

        # Preprocess
        inputs = preprocess_audio(waveform, self.sample_rate)

        # Get label
        angle_label = self.labels[idx]
        angle_label = torch.tensor(angle_label, dtype=torch.long)

        return inputs, angle_label

# 4. Define the Model Architecture

class AngleClassifier(nn.Module):
    def __init__(self, num_angle_bins):
        super(AngleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.gelu = nn.GELU()

        # Calculate number of patches based on input dimensions
        # Assuming input size: [4, 128, time_steps]
        # After conv1 and patch embedding with kernel_size and stride of (16, 16)
        self.num_patches_h = 8  # 128 / 16
        self.num_patches_w = None  # Need to calculate based on time_steps

        # Placeholder for time_steps after preprocessing
        example_waveform = torch.zeros(2, self.sample_rate)  # 1 second of audio
        Z = preprocess_audio(example_waveform, self.sample_rate)
        time_steps = Z.shape[2]

        # Calculate num_patches_w based on time_steps
        self.num_patches_w = time_steps // 16  # Assuming stride and kernel_size of 16

        num_patches = self.num_patches_h * self.num_patches_w

        # Patch Embedding
        self.patch_embed = nn.Conv2d(64, 768, kernel_size=(16, 16), stride=(16, 16))

        # Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, 768))

        # [CLS] token for direction
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))

        # Transformer Encoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # Output layer for direction classification
        self.fc_direction = nn.Linear(768, num_angle_bins)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor Z [batch_size, 4, mel_bins, time_steps].
        Returns:
            Tensor: Output logits for angle classification [batch_size, num_angle_bins].
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)

        # Patch embedding
        x = self.patch_embed(x)  # [batch_size, 768, num_patches_h, num_patches_w]
        x = x.flatten(2).transpose(1, 2)  # [batch_size, num_patches, 768]

        batch_size = x.size(0)
        num_patches = x.size(1)

        # Prepare [CLS] token
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, 768]

        # Concatenate [CLS] token and input tokens
        x = torch.cat((cls_token, x), dim=1)  # [batch_size, 1 + num_patches, 768]

        # Add positional encoding
        x = x + self.pos_embedding[:, : x.size(1), :]

        # Transformer Encoder
        x = self.transformer_encoder(x)  # batch_first=True

        # Extract output from [CLS] token
        direction_token = x[:, 0, :]  # [batch_size, embedding_dim]

        # Output layer
        direction_output = self.fc_direction(direction_token)  # [batch_size, num_angle_bins]

        return direction_output

# 5. Training the Model

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Prepare dataset and dataloader
dataset = SpatialAudioDataset(audio_files=audio_files, labels=angle_labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model, optimizer, and loss function
model = AngleClassifier(num_angle_bins=num_angle_bins)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 10  # Adjust the number of epochs as needed

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, angle_labels in dataloader:
        # Move data to device
        inputs = inputs.to(device)
        angle_labels = angle_labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        direction_output = model(inputs)

        # Compute loss
        loss = criterion(direction_output, angle_labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print epoch statistics
    avg_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 6. Evaluation (Optional)

# Evaluate the model on the training data or a separate validation set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, angle_labels in dataloader:
        inputs = inputs.to(device)
        angle_labels = angle_labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += angle_labels.size(0)
        correct += (predicted == angle_labels).sum().item()

accuracy = 100 * correct / total
print(f'Training Accuracy: {accuracy:.2f}%')
