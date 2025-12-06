import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load MNIST dataset directly from Keras (no need to download from Kaggle)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print("Dataset loaded successfully!")
print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Display sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()
for i in range(10):
    axes[i].imshow(X_train[i], cmap='gray')
    axes[i].set_title(f"Label: {y_train[i]}")
    axes[i].axis('off')
plt.suptitle('Sample Images from MNIST Dataset', fontsize=16)
plt.tight_layout()
plt.show()

# Data Preprocessing
# Normalize pixel values to range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape data to add channel dimension (for CNN)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode the labels
y_train_encoded = to_categorical(y_train, 10)
y_test_encoded = to_categorical(y_test, 10)

print(f"X_train shape after reshaping: {X_train.shape}")
print(f"X_test shape after reshaping: {X_test.shape}")
print(f"y_train_encoded shape: {y_train_encoded.shape}")

# Split training data into training and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train_encoded, test_size=0.1, random_state=42, stratify=y_train_encoded
)

print(f"Training set: {X_train_split.shape}")
print(f"Validation set: {X_val.shape}")

# Build CNN Model
model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Fully Connected Layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# Display model architecture
model.summary()

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)

# Add callbacks for better training
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
        min_lr=0.00001,
        verbose=1
    )
]

# Train the model
print("Training the CNN model...")
history = model.fit(
    X_train_split, y_train_split,
    epochs=50,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# Evaluate on test set
print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)
test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(
    X_test, y_test_encoded, verbose=0
)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Classification report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot accuracy
axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0, 0].set_title('Model Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot loss
axes[0, 1].plot(history.history['loss'], label='Training Loss')
axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
axes[0, 1].set_title('Model Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot precision
axes[1, 0].plot(history.history['precision'], label='Training Precision')
axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
axes[1, 0].set_title('Model Precision')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Plot recall
axes[1, 1].plot(history.history['recall'], label='Training Recall')
axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
axes[1, 1].set_title('Model Recall')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Visualize predictions on test samples
def visualize_predictions(num_samples=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # Get a random test sample
        idx = np.random.randint(0, len(X_test))
        test_image = X_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        
        # Make prediction
        pred_prob = model.predict(X_test[idx].reshape(1, 28, 28, 1), verbose=0)
        pred_label = np.argmax(pred_prob)
        confidence = np.max(pred_prob)
        
        # Display
        axes[i].imshow(test_image, cmap='gray')
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}')
        axes[i].axis('off')
        
        # Highlight incorrect predictions in red
        if true_label != pred_label:
            axes[i].spines['bottom'].set_color('red')
            axes[i].spines['top'].set_color('red')
            axes[i].spines['left'].set_color('red')
            axes[i].spines['right'].set_color('red')
            axes[i].spines['bottom'].set_linewidth(3)
            axes[i].spines['top'].set_linewidth(3)
            axes[i].spines['left'].set_linewidth(3)
            axes[i].spines['right'].set_linewidth(3)
    
    plt.suptitle('Sample Predictions on Test Set (Red border = Incorrect)', fontsize=16)
    plt.tight_layout()
    plt.show()

visualize_predictions(10)

# Error analysis: Show misclassified examples
def show_misclassified(num_samples=10):
    # Find misclassified examples
    misclassified_idx = np.where(y_pred != y_test)[0]
    
    if len(misclassified_idx) == 0:
        print("No misclassified examples found!")
        return
    
    print(f"Total misclassified: {len(misclassified_idx)}")
    
    # Display some misclassified examples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    num_samples = min(num_samples, len(misclassified_idx))
    for i in range(num_samples):
        idx = misclassified_idx[i]
        test_image = X_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        
        # Get prediction probabilities
        pred_prob = y_pred_prob[idx]
        
        axes[i].imshow(test_image, cmap='gray')
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nProb: {pred_prob[pred_label]:.2%}')
        axes[i].axis('off')
    
    plt.suptitle('Misclassified Examples', fontsize=16)
    plt.tight_layout()
    plt.show()

show_misclassified(10)

# Save the model
model.save('mnist_cnn_model.h5')
print("\nModel saved as 'mnist_cnn_model.h5'")

# Create a simpler model for comparison (optional)
def create_simple_cnn():
    simple_model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    simple_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return simple_model

# Train and evaluate simple model (optional)
print("\n" + "="*60)
print("TRAINING SIMPLE CNN FOR COMPARISON")
print("="*60)

simple_model = create_simple_cnn()
simple_history = simple_model.fit(
    X_train_split, y_train_split,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=0
)

simple_test_loss, simple_test_accuracy = simple_model.evaluate(X_test, y_test_encoded, verbose=0)
print(f"Simple CNN Test Accuracy: {simple_test_accuracy:.4f}")
print(f"Our CNN Test Accuracy: {test_accuracy:.4f}")
print(f"Improvement: {test_accuracy - simple_test_accuracy:.4f}")

# Model comparison visualization
plt.figure(figsize=(10, 6))
plt.plot(history.history['val_accuracy'], label='Our CNN Validation Accuracy', linewidth=2)
plt.plot(simple_history.history['val_accuracy'], label='Simple CNN Validation Accuracy', linewidth=2)
plt.title('Model Comparison: Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()