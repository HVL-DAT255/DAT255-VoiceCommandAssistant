import os
import sys
import argparse
import numpy as np
import tensorflow as tf

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocess import preprocess_audio

# Define your command classes (must match the order used during training)
COMMANDS = ["up", "down", "left", "right", "yes"]

# Path to the saved LSTM model (update the path if necessary)
MODEL_DIR = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/models"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_mfcc.keras")

def pad_or_truncate(mfcc: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Ensure MFCC has exactly `target_frames` frames (second dimension).

    Parameters
    ----------
    mfcc : np.ndarray
        Shape (40, T)       – output of preprocess_audio()
    target_frames : int
        Desired T (time‑steps) count – taken from model.input_shape[1]

    Returns
    -------
    np.ndarray
        Shape (40, target_frames)
    """
    n_mfcc, T = mfcc.shape
    if T < target_frames:                        # pad with zeros on the right
        pad_width = target_frames - T
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:                                        # or clip
        mfcc = mfcc[:, :target_frames]
    return mfcc


def predict_command(wav_path: str) -> str:
    """
    Predict the spoken command contained in `wav_path`.

    Returns the best class label as a string and prints the probabilities.

    Notes
    -----
    * Expects that the training script stored the model with a *fixed* time
      dimension (e.g. input shape (None, 63, 40)).  The helper above pads/
      truncates the incoming file to that length.
    """

    # 1) load model ----------------------------------------------------------
    model = tf.keras.models.load_model(MODEL_PATH)
    _, time_steps, n_mfcc = model.input_shape        # (None, T, 40)

    # 2) preprocess audio ----------------------------------------------------
    mfcc = preprocess_audio(wav_path)                # (40, T')
    if mfcc.shape[0] != n_mfcc:
        raise ValueError(
            f"Expected {n_mfcc} MFCC coefficients, got {mfcc.shape[0]}."
        )

    mfcc = pad_or_truncate(mfcc, target_frames=time_steps)   # (40, T)
    mfcc = mfcc.T                              # (T, 40) – time first
    mfcc = np.expand_dims(mfcc, axis=0)        # (1, T, 40) – batch dim

    # 3) predict -------------------------------------------------------------
    probs = model.predict(mfcc, verbose=0)[0]  # (n_classes,)
    pred_idx = int(np.argmax(probs))
    pred_cmd = COMMANDS[pred_idx]

    # 4) report --------------------------------------------------------------
    pretty = ", ".join(
        f"{cmd}: {prob:.2%}" for cmd, prob in zip(COMMANDS, probs)
    )
    print(f"\nFile : {os.path.basename(wav_path)}")
    print(f"Pred. : {pred_cmd}\nProbs : {pretty}\n")

    return pred_cmd


# ─── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict the spoken command in a WAV file"
    )
    parser.add_argument(
        "wav",
        type=str,
        help="Path to a .wav file to classify"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.wav):
        sys.exit(f"❌ File not found: {args.wav}")

    predict_command(args.wav)