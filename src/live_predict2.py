import os
import numpy as np
import tensorflow as tf
import sounddevice as sd
import librosa
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer 

class VoiceCommandRecognizer:
    """
    Records a short audio clip from the mic, extracts MFCCs, and
    predicts one of the predefined commands using a pretrained Keras model.
    """
    LABELS = ["up", "down", "left", "right", "yes"]
    DURATION = 1.5       # length of each recording (sec)
    SAMPLE_RATE = 16000  # sampling rate (Hz)

    def __init__(self,
                 model_dir: str,
                 model_filename: str = "lstm_mfcc.keras"):
        # where to find the .keras model on disk
        path = os.path.join(model_dir, model_filename)
        # load with custom_objects fix for Functional
        self._model = load_model(
            path,
            custom_objects={ "Functional": Model,
                            "InputLayer": InputLayer
                            },
            compile=False
        )

    def record(self) -> np.ndarray:
        """
        Records `DURATION` seconds from the default microphone
        at SAMPLE_RATE. Returns a 1-D float32 array.
        """
        print("🎙️  Listening now...")
        data = sd.rec(
            int(self.DURATION * self.SAMPLE_RATE),
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        return data.squeeze()

    def _extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Turn raw audio samples into a (1, 63, 40) MFCC batch
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.SAMPLE_RATE,
            n_mfcc=40
        )
        # fix to exactly 32 frames (time steps)
        mfcc = librosa.util.fix_length(mfcc, size=63, axis=1)
        # model expects (batch, time, features)
        return mfcc.T[np.newaxis, ...]

    def predict(self, audio: np.ndarray) -> str:
        """
        Given a raw audio array, return the top-1 command string.
        """
        batch = self._extract_mfcc(audio)
        probs = self._model.predict(batch)
        idx   = int(np.argmax(probs, axis=1)[0])
        return self.LABELS[idx]

    def listen_and_predict(self) -> str:
        """
        Convenience method: record, predict, and print.
        """
        audio   = self.record()
        command = self.predict(audio)
        print("🗣️  Predicted command:", command)
        return command


if __name__ == "__main__":
    # Quick smoke-test
    MODEL_DIR = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/models"
    recognizer = VoiceCommandRecognizer(MODEL_DIR)
    recognizer.listen_and_predict()
