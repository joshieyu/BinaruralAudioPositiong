import sounddevice as sd

# List all audio devices
print(sd.query_devices())


# Record with specified input device (e.g., microphones)
# recording = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=CHANNELS, dtype=np.float32, device=5)
