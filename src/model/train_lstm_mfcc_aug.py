#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────
# Train an LSTM‑based (or CNN) audio‑command classifier on any
# features exported by preprocess.py (mfcc, mfcc_aug, log_mel, …).
# ────────────────────────────────────────────────────────────────────

import argparse, os, sys, numpy as np, tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization,
                                     Bidirectional, LSTM, Dense, Dropout,
                                     GlobalAveragePooling1D)
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# ---------------- edit your folders here ---------------- #
ROOT         = Path(__file__).resolve().parents[2]          # project root
PROCESSED_DIR= ROOT / "data" / "processed"
MODEL_DIR    = ROOT / "models"
# -------------------------------------------------------- #

# ───────────────────────────────────────────────────────── #
# Data loader
# ───────────────────────────────────────────────────────── #
def load_data(method: str):
    """Load X_<method>.npy & y_<method>.npy and prep for the network."""
    X = np.load(PROCESSED_DIR / f"X_{method}.npy")
    y = np.load(PROCESSED_DIR / f"y_{method}.npy")

    # (samples, n_feats, n_frames) → (samples, n_frames, n_feats)
    X = np.transpose(X, (0, 2, 1))

    n_classes = int(y.max() + 1)
    y_cat     = tf.keras.utils.to_categorical(y, num_classes=n_classes)
    return X.astype("float32"), y_cat.astype("float32"), n_classes


# ───────────────────────────────────────────────────────── #
# Architectures
# ───────────────────────────────────────────────────────── #
def lstm_model(input_shape, n_classes,
               lstm1=128, lstm2=64, drop=0.3, conv_filters=64) -> Model:
    """Conv1D → Bi‑LSTM → Dropout → Bi‑LSTM → Dense."""
    inp = Input(shape=input_shape)

    x = Conv1D(conv_filters, 3, padding="same", activation="relu")(inp)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(lstm1, return_sequences=True))(x)
    x = Dropout(drop)(x)
    x = Bidirectional(LSTM(lstm2))(x)
    x = Dropout(drop)(x)

    x = Dense(32, activation="relu")(x)
    out= Dense(n_classes, activation="softmax")(x)

    model = Model(inp, out, name="LSTM_net")
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def cnn_model(input_shape, n_classes,
              filters=(64,128,128), drop=0.3) -> Model:
    """Pure 1‑D CNN baseline for comparison."""
    inp = Input(shape=input_shape)
    x   = inp
    for f in filters:
        x = Conv1D(f, 3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(drop)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    out= Dense(n_classes, activation="softmax")(x)
    model = Model(inp, out, name="CNN_net")
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ───────────────────────────────────────────────────────── #
# Optional Keras‑Tuner search  (very small search space demo)
# ───────────────────────────────────────────────────────── #
def tune_hyperparams(X_train, y_train, input_shape, n_classes):
    import keras_tuner as kt

    def model_builder(hp):
        lstm1 = hp.Int("lstm1",  64, 160, step=32)
        lstm2 = hp.Int("lstm2",  32, 128, step=32)
        drop  = hp.Float("drop", 0.1, 0.5, step=0.1)
        return lstm_model(input_shape, n_classes, lstm1, lstm2, drop)

    tuner = kt.RandomSearch(
        model_builder,
        objective="val_accuracy",
        max_trials=20,
        directory="kt_search",
        overwrite=True,
    )
    tuner.search(X_train, y_train, epochs=20, validation_split=0.2, verbose=0)
    return tuner.get_best_models(1)[0]


# ───────────────────────────────────────────────────────── #
def train(method: str, do_tune: bool):

    # 1) data
    X, y, n_classes = load_data(method)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=np.argmax(y, axis=1)
    )

    # 2) class weights
    cw = class_weight.compute_class_weight(
            "balanced",
            classes=np.unique(np.argmax(y_train, axis=1)),
            y=np.argmax(y_train, axis=1))
    cw = dict(enumerate(cw))

    # 3) pick &/or tune a model
    if do_tune:
        model = tune_hyperparams(X_train, y_train, X_train.shape[1:], n_classes)
        print("Best hyper‑parameters:", model.get_config()["layers"][3]["config"])
    else:
        model = lstm_model(X_train.shape[1:], n_classes)

    # 4) callbacks
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
    ]

    # 5) train
    model.fit(X_train, y_train,
              epochs=50, batch_size=32, validation_split=0.2,
              class_weight=cw, callbacks=cb, verbose=2)

    # 6) evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[{model.name}] Test  acc={acc:.4f}  loss={loss:.4f}")

    preds  = np.argmax(model.predict(X_test, verbose=0), axis=1)
    truths = np.argmax(y_test, axis=1)
    print(confusion_matrix(truths, preds))
    print(classification_report(truths, preds))

    # 7) save
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    out = MODEL_DIR / f"{model.name.lower()}_{method}.keras"
    model.save(out)
    print("✓ Saved to", out)


# ───────────────────────────────────────────────────────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="mfcc",
                        help="which feature set to load (mfcc, mfcc_aug …)")
    parser.add_argument("--tune", action="store_true",
                        help="run a random‑search hyper‑parameter tuning first")
    args = parser.parse_args()

    # quick comparison: LSTM vs CNN (optional)
    train(args.method, do_tune=args.tune)        # LSTM (optionally tuned)
    train(args.method, do_tune=False)            # CNN baseline
