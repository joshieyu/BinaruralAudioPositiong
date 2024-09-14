import os
import time
import wave
import pyaudio
import soundfile as sf
from glob import glob
import threading

# Configuration
sound_folder = "Sounds/"  # Folder with audio files to play
output_folder = "recordings/"  # Folder where recordings will be saved
directions = ["front", "back", "left", "right"]  # Direction bins
sample_rate = 48000  # Sampling rate for recording
channels = 2  # Stereo recording
record_seconds = 5  # Length of each recording in seconds (adjust as needed)

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Audio capture setup using pyaudio
p = pyaudio.PyAudio()

def record_audio(filename, duration):
    """Record stereo audio and save to a file while the sound plays."""
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=sample_rate, input=True, input_device_index=3,frames_per_buffer=1024)

    frames = []
    print(f"Recording for {duration} seconds...")
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("Recording complete.")
    stream.stop_stream()
    stream.close()

    # Save the recorded frames as a .wav file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def play_sound(file_path):
    """Play a sound file through the default audio output device."""
    data, samplerate = sf.read(file_path)
    sf.write('temp_output.wav', data, samplerate)
    os.system("afplay temp_output.wav")  # MacOS-specific, use "aplay" for Linux, or "start" for Windows

def play_and_record(sound_file, output_filename, duration):
    """Play the sound and record audio simultaneously."""
    # Start recording in a separate thread
    record_thread = threading.Thread(target=record_audio, args=(output_filename, duration))
    record_thread.start()

    # Play the sound
    play_sound(sound_file)

    # Wait for recording to finish
    record_thread.join()

def process_direction(direction):
    """Process all sounds for a given direction."""
    # Create a folder for this direction if it doesn't exist
    direction_folder = os.path.join(output_folder, direction)
    os.makedirs(direction_folder, exist_ok=True)

    # Get all sound files
    sound_files = glob(os.path.join(sound_folder, "*.wav"))

    for idx, sound_file in enumerate(sound_files):
        print(f"Playing sound {idx + 1}/{len(sound_files)} for direction '{direction}'")
        
        # Set output filename
        output_filename = os.path.join(direction_folder, f"{os.path.basename(sound_file)}_recording.wav")

        # Play and record simultaneously
        play_and_record(sound_file, output_filename, record_seconds)

        time.sleep(1)  # Pause between each recording

    print(f"All sounds processed for direction '{direction}'.")

def main():
    for direction in directions:
        print(f"Please place the speaker in the '{direction}' position and press Enter to continue...")
        input()  # Wait for the user to confirm speaker placement
        process_direction(direction)
        print(f"All sounds for direction '{direction}' are done. Please move the speaker for the next direction.")

    print("Sound collection completed for all directions!")

if __name__ == "__main__":
    main()

# Close pyaudio when done
p.terminate()
