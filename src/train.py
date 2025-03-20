import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load preprocessed features and labels
X = np.load("/Users/sondrerisnes/Documents/GitHub/DAT255-VoiceCommandAssistant/data/processed/X.npy")
y = np.load("/Users/sondrerisnes/Documents/GitHub/DAT255-VoiceCommandAssistant/data/processed/y.npy")

# Reshape for CNN (expand channel dimension)
X = np.expand_dims(X, axis=-1)  # Shape: (num_samples, 40, time_steps, 1)

# Split into training (80%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
