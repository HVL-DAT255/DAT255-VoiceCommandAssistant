import os
import sys
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
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")

def predict_command(file_path):
    """
    Load a trained model, preprocess an audio file, and predict the spoken command.
    :param file_path: Path to the audio file to be processed.
    :return: Predicted command as a string.
    """
    # Load the trained LSTM model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Preprocess the audio file.
    # preprocess_audio returns MFCC features as a 2D array of shape (40, T)
    mfcc = preprocess_audio(file_path)
    
    # Transpose MFCC features so that the time axis comes first.
    # Training code transposed to (T, 40)
    mfcc = mfcc.T

    # Add a batch dimension so that input shape becomes (1, T, 40)
    mfcc = np.expand_dims(mfcc, axis=0)
    
    # Get the prediction probabilities and then the predicted class index
    pred_probs = model.predict(mfcc)
    pred_index = np.argmax(pred_probs, axis=1)[0]
    
    # Map the index to the corresponding command
    predicted_command = COMMANDS[pred_index]
    print("Predicted command:", predicted_command)
    return predicted_command

if __name__ == "__main__":
    # Replace with the path to an actual audio file from your raw data directory
    test_file_path = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/data/raw/up/0ac15fe9_nohash_0.wav"
    predict_command(test_file_path)
