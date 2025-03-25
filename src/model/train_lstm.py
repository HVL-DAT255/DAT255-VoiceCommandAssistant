import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

#path to processed data
PROCESSED_DIR = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/data/processed"

def load_data():
    """ Load preprocessed data """
    X_path = os.path.join(PROCESSED_DIR, "X.npy")
    y_path = os.path.join(PROCESSED_DIR, "y.npy")

    X = np.load(X_path)
    y = np.load(y_path)

    #transpose MFCC to (NUM_SAMPLES, NUM_TIMESTEPS, NUM_FEATURES)
    X = np.transpose(X, (0, 2, 1))

    #Convert labels to one-hot
    num_classes = len(np.unique(y))
    y_cat = tf.keras.utils.to_categorical(y, num_classes)

    return X, y_cat, num_classes

def build_lstm_model(input_shape, num_classes):
    """ Build LSTM model """

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax")

    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_save_model():
    """ Load data, build LSTM model, train and save model """

    #1. load data
    X, y_cat, num_classes = load_data()
    print("data loaded. X shape: ", X.shape, "y shape: ", y_cat.shape)

    #2. build model
    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]), num_classes=num_classes)
    model.summary()

    #3. train model
    history = model.fit(X, y_cat, epochs=20, batch_size=32, validation_split=0.2)

    #4. save the trained model
    os.makedirs(os.path.join(PROCESSED_DIR, "models"), exist_ok=True)
    model_path = os.path.join(PROCESSED_DIR, "models", "lstm_model.h5")
    model.save(model_path)
    print("Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()

    