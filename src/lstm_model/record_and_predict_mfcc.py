import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import os, sys, time

MODEL_PATH = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/models/lstm_mfcc.keras"
LABELS = ["up", "down", "left", "right", "yes"] 

# ─── Audio & MFCC parameters  (MUST match training!) ─────────────────────────
SAMPLE_RATE = 16_000     # 16‑kHz mono
DURATION    = 1.5        # seconds to record

N_MFCC      = 40         # number of MFCC coefficients (training value)
N_FFT       = 512        # FFT window size        (training value)
HOP_LENGTH  = 256        # hop length / stride    (training value)

# ─── Load model and read input shape ─────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_PATH)
_, time_steps, n_mfcc = model.input_shape      # (None, T, F)
assert n_mfcc == N_MFCC, (
    f"Model expects {n_mfcc} MFCCs but N_MFCC={N_MFCC}. "
    "Fix N_MFCC or retrain."
)

# ──────────────────────────────────────────────────────────────────────────────
def record_clip():
    
    print("🎙️  Listening… Speak now!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype="float32")
    sd.wait()
    print("🛑 Stopped recording.")
    return np.squeeze(audio)

def mfcc_from_audio(y):
   
    m = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    m = librosa.util.fix_length(m, size=time_steps, axis=1)
    m = m.T                        # (T, N_MFCC)
    m = np.expand_dims(m, axis=0)  # add batch dim → (1, T, N_MFCC)
    return m.astype("float32")

def predict_command():
    y = record_clip()
    mfcc = mfcc_from_audio(y)
    probs = model.predict(mfcc, verbose=0)[0]
    idx   = int(np.argmax(probs))
    cmd   = LABELS[idx]
    prob  = probs[idx] * 100
    # Nicely print probabilities
    prob_str = ", ".join(f"{l}: {p*100:.1f}%" for l, p in zip(LABELS, probs))
    print(f"\n🧠  Predicted: {cmd}  ({prob:.1f}%)")
    print(f"🔎  All probs : {prob_str}\n")
    return cmd

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        while True:
            predict_command()
            time.sleep(0.5)
            print("Press Ctrl‑C to quit, or speak again…\n")
    except KeyboardInterrupt:
        print("👋  Bye!")