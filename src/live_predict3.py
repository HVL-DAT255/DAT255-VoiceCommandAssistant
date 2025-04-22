#!/usr/bin/env python3
# src/model/live_predict_mfcc.py
"""
Audio‑helper for the Streamlit Tic‑Tac‑Toe game.
Records the microphone, extracts MFCCs exactly like during training,
and runs the lstm_mfcc.keras model.
"""

import os, sys
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf

# ─── Paths ───────────────────────────────────────────────────────────────────
MODEL_DIR = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/models"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_mfcc.keras")
LABELS = ["up", "down", "left", "right", "yes"]


# ─── Audio / MFCC params  (must match training) ──────────────────────────────
SAMPLE_RATE = 16_000     # Hz
DURATION    = 1.5        # seconds you want to record

N_FFT       = 512        # <── values you used when you created X_mfcc.npy
HOP_LENGTH  = 256

# ─── Load model once ─────────────────────────────────────────────────────────
voice_model = tf.keras.models.load_model(MODEL_PATH)
_, TIME_STEPS, N_MFCC = voice_model.input_shape
print(f"Model expects shape (None, {TIME_STEPS}, {N_MFCC})")

# ─────────────────────────────────────────────────────────────────────────────
def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    """Record one clip from the default microphone."""
    print("🎙️  Listening…")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio)

def extract_mfcc(y, sr=SAMPLE_RATE):
    """MFCC → pad / crop to TIME_STEPS, return shape (TIME_STEPS, N_MFCC)."""
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )                           # (N_MFCC, T_raw)
    mfcc = librosa.util.fix_length(mfcc, size=TIME_STEPS, axis=1)
    return mfcc.T               # → (TIME_STEPS, N_MFCC)

def predict_command(audio):
    """Return one of the strings in LABELS."""
    mfcc = extract_mfcc(audio)
    mfcc = np.expand_dims(mfcc, axis=0)           # (1, T, F)
    probs = voice_model.predict(mfcc, verbose=0)[0]
    return LABELS[int(np.argmax(probs))]

# Quick manual test
if __name__ == "__main__":
    wav = record_audio()
    print("Predicted:", predict_command(wav))
