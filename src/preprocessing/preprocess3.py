

from __future__ import annotations
import os
import librosa
import numpy as np


DATA_DIR     = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/data/raw"
OUTPUT_DIR   = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/data/processed"
COMMANDS     = ["up", "down", "left", "right", "yes"]      
os.makedirs(OUTPUT_DIR, exist_ok=True)


try:
    from src.preprocessing.augment import augment_waveform  
except ImportError:
    augment_waveform = None              



def preprocess_audio(
        file_path: str,
        *,
        method      : str = "mfcc",
        max_length  : int = 16_000,
        sr          : int = 16_000,
        n_mfcc      : int = 40,
        n_mels      : int = 64,
        n_fft       : int = 512,
        hop_length  : int = 256
) -> np.ndarray:
    
    
    y, _ = librosa.load(file_path, sr=sr)
    y = librosa.util.fix_length(y, size=max_length)

 
    if method == "mfcc_aug":
        if augment_waveform is None:
            raise RuntimeError(
                "augment_waveform() not available; cannot use method 'mfcc_aug'."
            )
        y = augment_waveform(y, sr=sr)
        method = "mfcc"            


    if method == "log_mel":
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        return librosa.power_to_db(S, ref=np.max)

    
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )

    if method == "mfcc":
        return mfcc
    if method == "deltas":
        d1 = librosa.feature.delta(mfcc, order=1)
        d2 = librosa.feature.delta(mfcc, order=2)
        return np.vstack([mfcc, d1, d2])

    raise ValueError(
        f"Unknown method {method!r}. "
        "Choose 'mfcc', 'mfcc_aug', 'log_mel' or 'deltas'."
    )



def save_features(method: str = "mfcc") -> None:
   
    X, y = [], []
    for label, command in enumerate(COMMANDS):
        folder = os.path.join(DATA_DIR, command)
        for fname in os.listdir(folder):
            if not fname.endswith(".wav"):
                continue
            path = os.path.join(folder, fname)
            feats = preprocess_audio(path, method=method)
            X.append(feats)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    np.save(os.path.join(OUTPUT_DIR, f"X_aug_{method}.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, f"y_aug_{method}.npy"), y)
    print(f"[{method}] Done. X shape = {X.shape}, saved to {OUTPUT_DIR}")



if __name__ == "__main__":
   
    save_features(method="mfcc")

