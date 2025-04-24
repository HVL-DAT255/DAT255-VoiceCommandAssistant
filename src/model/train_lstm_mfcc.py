#!/usr/bin/env python3
# src/model/train_lstm_mfcc.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization,
    Bidirectional, LSTM, Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ——— CONFIG —————————————————————————————————————————————————————————————
# Point these to wherever you've saved X_mfcc.npy / y_mfcc.npy
PROCESSED_DIR = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/data/processed"
MODEL_DIR     = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/models"
METHOD        = "mfcc"   # loads X_mfcc.npy / y_mfcc.npy
# ————————————————————————————————————————————————————————————————————————

def load_data(method=METHOD):
    """
    Loads X_<method>.npy and y_<method>.npy, transposes from
    (n_samples, n_feats, n_frames) → (n_samples, n_frames, n_feats),
    and one‑hot encodes the labels.
    """
    X = np.load(os.path.join(PROCESSED_DIR, f"X_{method}.npy"))
    y = np.load(os.path.join(PROCESSED_DIR, f"y_{method}.npy"))

    # swap (feat,frames) → (frames,feat)
    X = np.transpose(X, (0, 2, 1))

    n_classes = len(np.unique(y))
    y_cat     = tf.keras.utils.to_categorical(y, num_classes=n_classes)
    return X, y_cat, n_classes


def build_model(input_shape, n_classes):
    """
    Conv1D → BatchNorm → Bi-LSTM → Dropout → Bi-LSTM → Dropout → Dense → Softmax
    """
    inp = Input(shape=input_shape)
    x   = Conv1D(64, 3, padding="same", activation="relu")(inp)
    x   = BatchNormalization()(x)

    x   = Bidirectional(LSTM(128, return_sequences=True))(x)
    x   = Dropout(0.3)(x)
    x   = Bidirectional(LSTM(64))(x)
    x   = Dropout(0.3)(x)

    x   = Dense(32, activation="relu")(x)
    out = Dense(n_classes, activation="softmax")(x)

    model = Model(inp, out)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train():
    # 1) load
    X, y, n_classes = load_data()
    print(f"Loaded  X={X.shape}, y={y.shape}, classes={n_classes}")

    # 2) split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=np.argmax(y, axis=1)
    )
    print(f"Split → train={X_train.shape}, test={X_test.shape}")

    # 3) class‑weights
    y_int     = np.argmax(y_train, axis=1)
    cw_values = class_weight.compute_class_weight(
                    "balanced",
                    classes=np.unique(y_int),
                    y=y_int
                )
    class_wts = dict(enumerate(cw_values))
    print("Class weights:", class_wts)

    # 4) build & inspect
    model = build_model(input_shape=X_train.shape[1:], n_classes=n_classes)
    model.summary()

    # 5) callbacks
    es = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    rl = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5
    )

    # 6) train
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_wts,
        callbacks=[es, rl],
        verbose=2
    )

    # 7) eval
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss={loss:.4f}  acc={acc:.4f}")

    preds = np.argmax(model.predict(X_test), axis=1)
    truths= np.argmax(y_test, axis=1)

    print("Confusion matrix:\n", confusion_matrix(truths, preds))
    print("\nClassification report:\n", classification_report(truths, preds))

    # 8) save
    os.makedirs(MODEL_DIR, exist_ok=True)
    outpath = os.path.join(MODEL_DIR, f"lstm_{METHOD}.keras")
    model.save(outpath)
    print("Model saved to", outpath)


if __name__ == "__main__":
    train()
