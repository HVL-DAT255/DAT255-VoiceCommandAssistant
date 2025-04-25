import os
import sys
import numpy as np
import tensorflow as tf

# denne delen hjelper med å finne riktig pathen
current_dir = os.path.dirname(__file__)  
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

current_dir = os.path.dirname(__file__)  # .../src/model
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from preprocess import preprocess_audio

COMMANDS = ["up", "down", "left", "right", "yes"]

# finner riktig pathen av hvor modelen transformer_model.keras ligger.
MODEL_PATH = os.path.join(project_root, "models", "transformer_model.keras")

def predict_command(file_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    mfcc = preprocess_audio(file_path)
    mfcc = tf.image.resize(mfcc[None, ..., None], [40, 32])[0, ..., 0].numpy()
    mfcc = mfcc.T  

    mfcc = np.expand_dims(mfcc, axis=0)  
    pred_probs = model.predict(mfcc)
    pred_index = np.argmax(pred_probs, axis=1)[0]
    predicted_command = COMMANDS[pred_index]
    print("🔮 Predicted command:", predicted_command)
    return predicted_command

if __name__ == "__main__":
    # teste med en fil i data folder for "yes"
    test_file_path = os.path.join(project_root, "data", "raw", "yes", "0bd689d7_nohash_0.wav")
    predict_command(test_file_path)
