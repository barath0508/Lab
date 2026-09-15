import numpy as np
import matplotlib.pyplot as plt
from emnist import extract_training_samples, extract_test_samples
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam

# Load EMNIST data
X_train, y_train = extract_training_samples('byclass')
X_test, y_test = extract_test_samples('byclass')

# Reshape images
X_train = X_train.reshape(len(X_train), 784).astype('float32') / 255
X_test = X_test.reshape(len(X_test), 784).astype('float32') / 255

# Convert labels to categorical
y_train = to_categorical(y_train, 62)
y_test = to_categorical(y_test, 62)

# Create Neural Network
model = Sequential()
model.add(Dense(512, input_shape=(784,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(62, activation='softmax'))

# Compile
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=256,
                    validation_split=0.2)

# Test
score = model.evaluate(X_test, y_test)
print("Test Accuracy:", score[1])

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.show()
