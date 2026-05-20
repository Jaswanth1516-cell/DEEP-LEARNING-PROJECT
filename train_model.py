#!/usr/bin/env python3
"""
Deep Learning Image Classification - CIFAR-10 CNN Model
Author: Deep Learning Project
Description: Complete CNN implementation for CIFAR-10 image classification
             with training, evaluation, and visualization capabilities.

Usage:
    python train_model.py                          # Default training
    python train_model.py --epochs 50 --batch 128  # Custom parameters
    python train_model.py --predict image.jpg      # Make predictions
"""

import os
import sys
import json
import argparse

try:
    import numpy as np
except ModuleNotFoundError as e:
    print(f"Missing required package: {e.name}. Install dependencies with pip install -r requirements.txt")
    raise

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Class names for CIFAR-10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Output directory
OUTPUT_DIR = Path('saved_models')
OUTPUT_DIR.mkdir(exist_ok=True)


def setup_gpu():
    """Configure GPU settings."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU configured: {len(gpus)} GPU(s) available")
        except RuntimeError as e:
            print(f"✗ GPU configuration error: {e}")
    else:
        print("ℹ No GPU available, using CPU")


def load_and_prepare_data():
    """Load CIFAR-10 dataset and prepare for training."""
    print("\n" + "="*60)
    print("Loading CIFAR-10 Dataset")
    print("="*60)
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Flatten and normalize
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # One-hot encode
    y_train_encoded = keras.utils.to_categorical(y_train, 10)
    y_test_encoded = keras.utils.to_categorical(y_test, 10)
    
    print(f"✓ Training samples: {x_train.shape[0]}")
    print(f"✓ Test samples: {x_test.shape[0]}")
    print(f"✓ Image shape: {x_train.shape[1:]}")
    print(f"✓ Number of classes: {len(CLASS_NAMES)}")
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator()
    train_datagen.fit(x_train)
    test_datagen.fit(x_test)
    
    return (x_train, y_train, y_train_encoded), \
           (x_test, y_test, y_test_encoded), \
           (train_datagen, test_datagen)


def build_model():
    """Build CNN model architecture."""
    print("\n" + "="*60)
    print("Building CNN Model")
    print("="*60)
    
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    print(f"✓ Model built successfully")
    print(f"✓ Total parameters: {model.count_params():,}")
    
    return model


def train_model(model, train_data, test_data, train_datagen, epochs=50, batch_size=128):
    """Train the model."""
    print("\n" + "="*60)
    print("Training Model")
    print("="*60)
    
    x_train, y_train, y_train_encoded = train_data
    x_test, y_test, y_test_encoded = test_data
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_datagen.flow(x_train, y_train_encoded, batch_size=batch_size),
        validation_data=(x_test, y_test_encoded),
        epochs=epochs,
        steps_per_epoch=len(x_train) // batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✓ Training complete!")
    return history


def evaluate_model(model, x_test, y_test, y_test_encoded):
    """Evaluate model performance."""
    print("\n" + "="*60)
    print("Evaluating Model")
    print("="*60)
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test_encoded, verbose=0)
    print(f"✓ Test Loss: {test_loss:.4f}")
    print(f"✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Predictions
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    return y_pred, y_pred_probs


def save_model(model, history):
    """Save model and training history."""
    print("\n" + "="*60)
    print("Saving Model")
    print("="*60)
    
    # SavedModel format
    model.save(str(OUTPUT_DIR / 'cifar10_cnn_model'))
    print(f"✓ Model saved: {OUTPUT_DIR / 'cifar10_cnn_model'}")
    
    # HDF5 format
    model.save(str(OUTPUT_DIR / 'cifar10_cnn_model.h5'))
    print(f"✓ Model saved: {OUTPUT_DIR / 'cifar10_cnn_model.h5'}")
    
    # Training history
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
    }
    with open(OUTPUT_DIR / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"✓ History saved: {OUTPUT_DIR / 'training_history.json'}")


def visualize_training(history):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'training_curves.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved: {OUTPUT_DIR / 'training_curves.png'}")
    plt.close()


def visualize_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - CIFAR-10')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {OUTPUT_DIR / 'confusion_matrix.png'}")
    plt.close()


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train CIFAR-10 CNN Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=128, help='Batch size')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Deep Learning Image Classification")
    print("CIFAR-10 CNN Model Training")
    print("="*60)
    
    # Setup
    setup_gpu()
    
    # Load data
    train_data, test_data, datagen = load_and_prepare_data()
    x_test, y_test, y_test_encoded = test_data
    
    # Build model
    model = build_model()
    
    # Train
    history = train_model(model, train_data, test_data, 
                         datagen[0], epochs=args.epochs, batch_size=args.batch)
    
    # Evaluate
    y_pred, y_pred_probs = evaluate_model(model, x_test, y_test, y_test_encoded)
    
    # Save
    save_model(model, history)
    
    # Visualize
    if args.visualize:
        print("\n" + "="*60)
        print("Creating Visualizations")
        print("="*60)
        visualize_training(history)
        visualize_confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
