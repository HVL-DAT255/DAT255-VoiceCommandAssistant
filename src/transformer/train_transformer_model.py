import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocess import preprocess_audio  # This must return (40, T) MFCC

# Config
DATA_DIR = "data/raw"
LABELS = ["up", "down", "left", "right", "yes"]
MODEL_PATH = "models/transformer_model.keras"
SAMPLE_LENGTH = 32  # Time steps
NUM_MFCC = 40
NUM_CLASSES = len(LABELS)

# Load and process dataset
def load_dataset():
    X, y = [], []
    for label_index, label in enumerate(LABELS):
        folder = os.path.join(DATA_DIR, label)
        for filename in os.listdir(folder):
            if filename.endswith(".wav"):
                file_path = os.path.join(folder, filename)
                try:
                    mfcc = preprocess_audio(file_path)  # (40, T)
                    mfcc = tf.image.resize(mfcc[None, ..., None], [NUM_MFCC, SAMPLE_LENGTH])[0, ..., 0].numpy()
                    mfcc = mfcc.T  # shape: (T, 40)
                    X.append(mfcc)
                    y.append(label_index)
                except Exception as e:
                    print(f"⚠️ Failed to process {file_path}: {e}")
    return np.array(X), tf.keras.utils.to_categorical(y, NUM_CLASSES)

# Simple Transformer encoder block
def transformer_block(inputs, num_heads=2, ff_dim=64):
    attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(inputs, inputs)
    x = tf.keras.layers.Add()([inputs, attention])
    x = tf.keras.layers.LayerNormalization()(x)

    ffn = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ffn = tf.keras.layers.Dense(inputs.shape[-1])(ffn)
    x = tf.keras.layers.Add()([x, ffn])
    x = tf.keras.layers.LayerNormalization()(x)
    return x

# Build the model
def build_model(input_shape=(SAMPLE_LENGTH, NUM_MFCC)):
    inputs = tf.keras.Input(shape=input_shape)
    x = transformer_block(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

# Train & save
if __name__ == "__main__":
    print("📥 Loading dataset...")
    X, y = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🔧 Building model...")
    model = build_model()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print("🚀 Training...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"✅ Saved Transformer model to {MODEL_PATH}")
