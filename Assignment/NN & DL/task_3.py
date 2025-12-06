import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the dataset from the provided content
from io import StringIO
df = pd.read_csv(StringIO(file_content))

# Display basic info
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nColumns:", df.columns.tolist())

# 1.1 Create sentiment labels based on Rating
# Since Rating is a string with potential commas, convert to float
def convert_rating(rating):
    if pd.isna(rating):
        return None
    # Remove commas and convert to float
    try:
        return float(str(rating).replace(',', ''))
    except:
        return None

df['Rating_clean'] = df['Rating'].apply(convert_rating)

# Drop rows without ratings
df = df.dropna(subset=['Rating_clean'])

# Create binary sentiment labels (1 = Positive, 0 = Negative)
# Threshold: >= 6.5 = Positive, < 6.5 = Negative
df['sentiment'] = (df['Rating_clean'] >= 6.5).astype(int)

print(f"Dataset size after cleaning: {df.shape}")
print(f"Positive reviews: {df['sentiment'].sum()}")
print(f"Negative reviews: {len(df) - df['sentiment'].sum()}")
print(f"Positive percentage: {(df['sentiment'].sum()/len(df)*100):.2f}%")

# 1.2 Clean the review text
def clean_text(text):
    if pd.isna(text):
        return ""
    # Convert to string
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

df['Review_clean'] = df['Review'].apply(clean_text)

# 1.3 Split the data
X = df['Review_clean']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# 2.1 Tokenization
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 200

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

# Convert texts to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

print(f"Vocabulary size: {len(tokenizer.word_index)}")
print(f"Training data shape: {X_train_pad.shape}")
print(f"Testing data shape: {X_test_pad.shape}")

# 2.2 Prepare labels
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)


def build_rnn_model(vocab_size, embedding_dim=128, rnn_units=64):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
                  input_length=MAX_SEQUENCE_LENGTH),
        SimpleRNN(rnn_units, return_sequences=False),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

rnn_model = build_rnn_model(MAX_VOCAB_SIZE)
print("RNN Model Summary:")
rnn_model.summary()


def build_lstm_model(vocab_size, embedding_dim=128, lstm_units=64):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                  input_length=MAX_SEQUENCE_LENGTH),
        LSTM(lstm_units, return_sequences=False),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

lstm_model = build_lstm_model(MAX_VOCAB_SIZE)
print("\nLSTM Model Summary:")
lstm_model.summary()


def build_gru_model(vocab_size, embedding_dim=128, gru_units=64):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                  input_length=MAX_SEQUENCE_LENGTH),
        GRU(gru_units, return_sequences=False),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

gru_model = build_gru_model(MAX_VOCAB_SIZE)
print("\nGRU Model Summary:")
gru_model.summary()


# Training parameters
BATCH_SIZE = 32
EPOCHS = 15
VALIDATION_SPLIT = 0.1

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Train RNN Model
print("\n" + "="*50)
print("Training RNN Model...")
print("="*50)
rnn_history = rnn_model.fit(
    X_train_pad, y_train_np,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stopping],
    verbose=1
)

# Train LSTM Model
print("\n" + "="*50)
print("Training LSTM Model...")
print("="*50)
lstm_history = lstm_model.fit(
    X_train_pad, y_train_np,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stopping],
    verbose=1
)

# Train GRU Model
print("\n" + "="*50)
print("Training GRU Model...")
print("="*50)
gru_history = gru_model.fit(
    X_train_pad, y_train_np,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stopping],
    verbose=1
)


def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n{'='*60}")
    print(f"Evaluation Results for {model_name}")
    print(f"{'='*60}")
    
    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_prob
    }

# Evaluate all models
results = {}
results['RNN'] = evaluate_model(rnn_model, X_test_pad, y_test_np, 'RNN Model')
results['LSTM'] = evaluate_model(lstm_model, X_test_pad, y_test_np, 'LSTM Model')
results['GRU'] = evaluate_model(gru_model, X_test_pad, y_test_np, 'GRU Model')