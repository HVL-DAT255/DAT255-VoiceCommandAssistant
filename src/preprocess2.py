import librosa
import numpy as np
import os

DATA_DIR = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/data/raw"
COMMANDS = ["up", "down", "left", "right", "yes"]
OUTPUT_DIR = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_audio(file_path,
                     max_length=16000,
                     method="mfcc",
                     sr=16000,
                     n_mfcc=40,
                     n_mels=64,
                     n_fft=512,
                     hop_length=256):
    """
    Load & pad/trunc, then extract features according to method:
      - "mfcc"      : MFCC only (n_mfcc).
      - "log_mel"   : log‑Mel spectrogram (n_mels).
      - "deltas"    : MFCC + Δ + ΔΔ (n_mfcc).
    """
    y, _ = librosa.load(file_path, sr=sr)
    y = librosa.util.fix_length(y, size=max_length)

    if method == "log_mel":
        # Mel power → dB
        S = librosa.feature.melspectrogram(y=y, sr=sr,
                                           n_mels=n_mels,
                                           n_fft=n_fft,
                                           hop_length=hop_length)
        return librosa.power_to_db(S, ref=np.max)

    # always compute MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=n_fft,
                                hop_length=hop_length)

    if method == "mfcc":
        return mfcc

    elif method == "deltas":
        d1 = librosa.feature.delta(mfcc, order=1)
        d2 = librosa.feature.delta(mfcc, order=2)
        return np.vstack([mfcc, d1, d2])

    else:
        raise ValueError(f"Unknown method {method!r}. Choose 'mfcc','log_mel' or 'deltas'.")

def save_features(method="mfcc"):
    """
    Process all audio files with the selected method and save X.npy, y.npy.
    """
    X, y = [], []
    for label, command in enumerate(COMMANDS):
        path = os.path.join(DATA_DIR, command)
        for fname in os.listdir(path):
            file_path = os.path.join(path, fname)
            feats = preprocess_audio(file_path, method=method)
            X.append(feats)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    np.save(os.path.join(OUTPUT_DIR, f"X_{method}.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, f"y_{method}.npy"), y)
    print(f"[{method}] Preprocessing complete! X.shape={X.shape}")

if __name__ == "__main__":
    # Pick one:
    save_features(method="mfcc")
    # save_features(method="log_mel")
    # save_features(method="deltas")