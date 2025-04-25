import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Bidirectional, LSTM, Dense, Dropout
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Paths to processed data and model directory
PROCESSED_DIR = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/data/processed"
MODEL_DIR = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/models"

def load_data():
    
    X_path = os.path.join(PROCESSED_DIR, "X.npy")
    y_path = os.path.join(PROCESSED_DIR, "y.npy")
    
    X = np.load(X_path)
    y = np.load(y_path)
    

    X = np.transpose(X, (0, 2, 1))
    
    num_classes = len(np.unique(y))
    y_cat = tf.keras.utils.to_categorical(y, num_classes)
    
    return X, y_cat, num_classes

def build_lstm_model(input_shape, num_classes):
   
    inputs = Input(shape=input_shape)

    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def augment_data(X, noise_factor=0.05):
   
    # Generate noise with the same shape as X
    noise = noise_factor * np.random.randn(*X.shape)
    return X + noise

def train_and_save_model():

    # 1. Load data
    X, y_cat, num_classes = load_data()
    print("Data loaded. X shape:", X.shape, "y shape:", y_cat.shape)
    
    # Create a train/test split (20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
    print("Train data:", X_train.shape, y_train.shape)
    print("Test data:", X_test.shape, y_test.shape)
    
    # 2. Compute class weights (to help with any imbalanced classes)
    y_train_integers = np.argmax(y_train, axis=1)
    weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train_integers), y=y_train_integers)
    class_weights_dict = dict(enumerate(weights))
    print("Class weights:", class_weights_dict)
    
    # 3. Augment the training data: concatenate original and augmented data
    noise_factor = 0.05  # adjust this factor as needed
    X_train_aug = np.concatenate([X_train, augment_data(X_train, noise_factor=noise_factor)], axis=0)
    y_train_aug = np.concatenate([y_train, y_train], axis=0)
    print("Augmented training data shape:", X_train_aug.shape, y_train_aug.shape)
    
    # 4. Build the improved model using the shape of the training data
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes)
    model.summary()
    
    # 5. Setup callbacks: EarlyStopping and ReduceLROnPlateau
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    
    # Train the model with validation split and class weights (on the augmented training data)
    history = model.fit(
        X_train_aug, y_train_aug,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights_dict,
        callbacks=[early_stopping, reduce_lr]
    )
    
    # 6. Evaluate on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}\n")
    
    # Generate confusion matrix and classification report
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)
    
    print("\nClassification report:")
    print(classification_report(y_true, y_pred))
    
    # 7. Save the model using the native Keras format
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "lstm_model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()