import os
import numpy as np
import tensorflow as tf
import sounddevice as sd
import librosa

# Configuration: update these paths if needed
MODEL_DIR_VOICE = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/models"
MODEL_PATH_VOICE = os.path.join(MODEL_DIR_VOICE, "lstm_model.keras")
LABELS = ["up", "down", "left", "right", "yes"]

# Audio configuration
DURATION = 1.5   # seconds
SAMPLE_RATE = 16000  # Hz

# Load the pretrained voice command model once
voice_model = tf.keras.models.load_model(MODEL_PATH_VOICE)

def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    """Record audio from the microphone for a specified duration."""
    print("🎙️ Listening... Speak now!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording finishes
    return np.squeeze(audio)

def extract_mfcc(y, sr=SAMPLE_RATE):
    """Extract MFCC features from an audio signal and fix the time axis to 32 frames."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = librosa.util.fix_length(mfcc, size=32, axis=1)
    return mfcc

def predict_command(audio):
    """
    Use the pretrained voice model to predict the spoken command.
    Returns a string from LABELS.
    """
    mfcc = extract_mfcc(audio)
    mfcc = mfcc.T  # Transpose to (32, 40): 32 time steps, 40 features
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension: (1, 32, 40)
    pred_probs = voice_model.predict(mfcc)
    pred_index = np.argmax(pred_probs, axis=1)[0]
    return LABELS[pred_index]

# For testing purposes:
if __name__ == "__main__":
    audio = record_audio()
    command = predict_command(audio)
    print("Predicted command:", command)
