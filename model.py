import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load data
inputs = []
outputs = []
with open("training_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        inputs.append(item["input"])
        outputs.append(item["output"])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(outputs)

# Tokenize text
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(inputs)
sequences = tokenizer.texts_to_sequences(inputs)
X = pad_sequences(sequences, maxlen=12, padding='post')

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build improved model
model = Sequential([
    Embedding(input_dim=1000, output_dim=32, input_length=12),
    Bidirectional(LSTM(16, return_sequences=False)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=2
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.2f}")

# Example prediction
sample = ["failed to connect to instance"]
sample_seq = tokenizer.texts_to_sequences(sample)
sample_pad = pad_sequences(sample_seq, maxlen=12, padding='post')
pred = model.predict(sample_pad)
pred_label = label_encoder.inverse_transform([int(pred[0][0] > 0.5)])
print(f"Prediction for '{sample[0]}': {pred_label[0]}")
