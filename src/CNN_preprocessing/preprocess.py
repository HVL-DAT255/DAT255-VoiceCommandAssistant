import librosa
import librosa.display
import numpy as np
import os
import tensorflow as tf

DATA_DIR = "/Users/sondrerisnes/Documents/GitHub/DAT255-VoiceCommandAssistant/data/raw"
COMMANDS = ["up", "down", "left", "right", "yes"]
OUTPUT_DIR = "/Users/sondrerisnes/Documents/GitHub/DAT255-VoiceCommandAssistant/data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_audio(file_path, max_length=16000):
    """ Load an audio file, pad or truncate, and extract MFCC features """
    y, sr = librosa.load(file_path, sr=16000)  
    y = librosa.util.fix_length(y, size=max_length)  
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  
    return mfcc

def save_features():
    """ Process all audio files and save as numpy arrays """
    X, y = [], []
    
    for label, command in enumerate(COMMANDS):
        command_path = os.path.join(DATA_DIR, command)
        for file in os.listdir(command_path):
            file_path = os.path.join(command_path, file)
            mfcc = preprocess_audio(file_path)
            X.append(mfcc)
            y.append(label)
    
    # Convert to numpy and save
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), np.array(X))
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), np.array(y))
    print("Preprocessing complete! Data saved.")

if __name__ == "__main__":
    save_features()