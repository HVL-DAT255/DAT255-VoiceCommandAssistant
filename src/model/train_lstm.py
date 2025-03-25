import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight


#path to processed data
PROCESSED_DIR = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/data/processed"
MODEL_DIR = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/models"

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

    #create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    print("Train data: ", X_train.shape, y_train.shape)
    print("Test data: ", X_test.shape, y_test.shape)

    #Compute class weights to help improve recall
    y_train_integers = np.argmax(y_train, axis=1)  # Convert one-hot labels back to integers
    weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train_integers), y=y_train_integers)
    class_weights_dict = dict(enumerate(weights))
    print("Class weights:", class_weights_dict)

    #2. build model
    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]), num_classes=num_classes)
    model.summary()

    #3. train model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, class_weight=class_weights_dict)

    #4. evaulate on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}\n")

    #5. confusion matrix and classification report
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix: ")
    print(cm)

    print("\nClassification report: ")
    print(classification_report(y_true, y_pred))


    #6. save the trained model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "lstm_model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()

