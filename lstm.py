from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

y_train_onehot = to_categorical(y_train, num_classes=3)
y_test_onehot = to_categorical(y_test, num_classes=3)

X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

model = Sequential([
    LSTM(64, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(0.01), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(
    X_train_reshaped, 
    y_train_onehot,
    epochs=100,
    batch_size=8,
    validation_data=(X_test_reshaped, y_test_onehot),
    verbose=1
)

loss, accuracy = model.evaluate(X_test_reshaped, y_test_onehot, verbose=0)
print(f"Final Test Accuracy: {accuracy:.4f}")

model.save("best_lstm_diabetes_model.keras")
print("Model saved as 'best_lstm_diabetes_model.keras'")

with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
