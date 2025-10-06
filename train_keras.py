# src/train_keras.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 1) Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2) Preprocess
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0
x_train = np.expand_dims(x_train, -1)  # (N,28,28,1)
x_test  = np.expand_dims(x_test, -1)

# 3) Build model
def build_model():
    inp = layers.Input(shape=(28,28,1))
    x = layers.Conv2D(32, (3,3), activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    return model

model = build_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# 4) Callbacks
cb = [
    callbacks.ModelCheckpoint('best_mnist_keras.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
    callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# 5) Train
history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=128,
                    validation_split=0.1,
                    callbacks=cb)

# 6) Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)

# 7) Predictions and confusion matrix
y_pred = np.argmax(model.predict(x_test), axis=1)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion matrix')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.colorbar()
plt.show()

# Save final model (SavedModel format)
model.save("mnist_cnn.keras")

