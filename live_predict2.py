import os
import numpy as np
import tensorflow as tf
import sounddevice as sd
import librosa
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer

class VoiceCommandRecognizer:
    LABELS      = ["up","down","left","right","yes"]
    DURATION    = 1.5
    SAMPLE_RATE = 16000

    def __init__(self, model_dir: str, model_filename: str = "lstm_mfcc.keras"):
        path = os.path.join(model_dir, model_filename)
        self._model = load_model(
            path,
            custom_objects={"Functional": Model, "InputLayer": InputLayer},
            compile=False
        )

    def _extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        mfcc = librosa.feature.mfcc(y=audio, sr=self.SAMPLE_RATE, n_mfcc=40)
        mfcc = librosa.util.fix_length(mfcc, size=63, axis=1)
        return mfcc.T[np.newaxis, ...]

    def predict(self, audio: np.ndarray) -> str:
        batch = self._extract_mfcc(audio)
        probs  = self._model.predict(batch)
        idx    = int(np.argmax(probs, axis=1)[0])
        return self.LABELS[idx]

    def record(self) -> np.ndarray:
        data = sd.rec(int(self.DURATION * self.SAMPLE_RATE),
                      samplerate=self.SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        return data.squeeze()

    def listen_and_predict(self) -> str:
        audio   = self.record()
        command = self.predict(audio)
        return command