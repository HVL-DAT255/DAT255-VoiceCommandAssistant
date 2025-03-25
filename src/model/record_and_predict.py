import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import os

MODEL_PATH = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/models/lstm_model.keras"
LABELS = ["up", "down", "left", "right", "yes"] 

# Audio config
DURATION = 1.5  # seconds
SAMPLE_RATE = 16000

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    print("🎙️ Listening... Speak now!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return np.squeeze(audio)

def extract_mfcc(y, sr=SAMPLE_RATE):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = librosa.util.fix_length(mfcc, size=32, axis=1)
    return mfcc

def predict_command(audio):
    # Extract MFCC features (shape will be (40, 32))
    mfcc = extract_mfcc(audio)
    # Transpose so that time axis comes first: (32, 40)
    mfcc = mfcc.T
    # Add batch dimension to get shape (1, 32, 40)
    mfcc = np.expand_dims(mfcc, axis=0)
    # Get prediction probabilities from the model
    prediction = model.predict(mfcc)
    pred_index = np.argmax(prediction)
    return LABELS[pred_index]

if __name__ == "__main__":
    audio = record_audio()
    command = predict_command(audio)
    print(f"🧠 Predicted command: {command}")
