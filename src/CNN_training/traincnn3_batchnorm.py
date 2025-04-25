import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import classification_report

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")
X = np.expand_dims(X, axis=-1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN with different batch normalization placement
def build_cnn_with_bn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Train model
model = build_cnn_with_bn(input_shape=X.shape[1:], num_classes=len(set(y)))
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping]
)

# Save and evaluate
model.save("models/speech_cnn_batchnorm.h5")
print("Model training complete! Saved at 'models/speech_cnn_batchnorm.h5'.")

y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(y_val, y_pred_classes, target_names=["up", "down", "left", "right", "yes"]))
