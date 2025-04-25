import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import os

# denne delen hjelper med å finne riktig pathen
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "transformer_model.keras")
LABELS = ["up", "down", "left", "right", "yes"]

# settingen for audioen
DURATION = 1.5  # i sekunder
SAMPLE_RATE = 16000
NUM_MFCC = 40
SAMPLE_LENGTH = 32

# laster in modell man skal bruke, i dette tilfelle er det transformer modellen
model = tf.keras.models.load_model(MODEL_PATH)


sd.default.device = 2

def record_audio(duration=DURATION, sr=SAMPLE_RATE, device=None):
    if device is not None:
        sd.default.device = device
    print("🎙️ Listening... Speak now!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return np.squeeze(audio)

def extract_mfcc(y, sr=SAMPLE_RATE):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=NUM_MFCC)
    mfcc = librosa.util.fix_length(mfcc, size=SAMPLE_LENGTH, axis=1)
    return mfcc

def predict_command(audio):
    audio = librosa.effects.trim(audio, top_db=20)[0]  # clean silence
    mfcc = extract_mfcc(audio)                         # (40, 32)
    mfcc = mfcc.T                                       # (32, 40)
    mfcc = np.expand_dims(mfcc, axis=0)                # (1, 32, 40)
    prediction = model.predict(mfcc)
    pred_index = np.argmax(prediction)
    return LABELS[pred_index]

if __name__ == "__main__":
    audio = record_audio()
    command = predict_command(audio)
    print(f"🧠 Predicted command: {command}")
