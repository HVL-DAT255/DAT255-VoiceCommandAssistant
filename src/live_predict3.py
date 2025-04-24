#!/usr/bin/env python3
# src/live_predict3.py

"""
Audio-helper for the Streamlit Tic-Tac-Toe game.
Records from the microphone, extracts MFCCs exactly like during training,
and runs the lstm_mfcc.keras model.
"""

import os
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.expanduser(
    "~/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/"
    "DAT255-VoiceCommandAssistant/models"
)
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_net_mfcc.keras")
LABELS = ["up", "down", "left", "right", "yes"]

# ─── Audio / MFCC params  (must match training) ───────────────────────────────
SAMPLE_RATE = 16_000    # Hz
DURATION    = 1.5       # seconds to record
N_FFT       = 512       # must match your training code
HOP_LENGTH  = 256

# ─── Patch InputLayer to accept old `batch_shape` in config ────────────────────
import keras.src.engine.input_layer as _il
_orig_init = _il.InputLayer.__init__
def _patched_init(self, *args, batch_shape=None, **kwargs):
    # Keras used to serialize InputLayers with `batch_shape=[None, ...]`
    # the new API expects `input_shape=(...,)`
    if batch_shape is not None:
        kwargs["input_shape"] = tuple(batch_shape[1:])
    return _orig_init(self, *args, **kwargs)
_il.InputLayer.__init__ = _patched_init

# ─── Load model once ──────────────────────────────────────────────────────────
voice_model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={"Functional": tf.keras.Model}
)
_, TIME_STEPS, N_MFCC = voice_model.input_shape
print(f"Model expects shape (None, {TIME_STEPS}, {N_MFCC})")

# ─── Utility functions ───────────────────────────────────────────────────────
def record_audio(duration=DURATION, sr=SAMPLE_RATE) -> np.ndarray:
    """Record one clip from the default microphone."""
    print("🎙️  Listening…")
    audio = sd.rec(
        int(duration * sr),
        samplerate=sr,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    return np.squeeze(audio)

def extract_mfcc(y: np.ndarray, sr=SAMPLE_RATE) -> np.ndarray:
    """MFCC → pad/crop to TIME_STEPS → shape (TIME_STEPS, N_MFCC)."""
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )                           # (n_mfcc, t_raw)
    mfcc = librosa.util.fix_length(mfcc, size=TIME_STEPS, axis=1)
    return mfcc.T               # → (TIME_STEPS, N_MFCC)

def predict_command(audio: np.ndarray) -> str:
    """Return one of the LABELS for a single audio clip."""
    mfcc = extract_mfcc(audio)                 # (TIME_STEPS, N_MFCC)
    mfcc = np.expand_dims(mfcc, axis=0)        # (1, TIME_STEPS, N_MFCC)
    probs = voice_model.predict(mfcc, verbose=0)[0]
    return LABELS[int(np.argmax(probs))]

# ─── Quick manual test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    wav = record_audio()
    print("Predicted:", predict_command(wav))
