import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# Data augmentation pour signaux 1D
def augment_signal(x, noise_level=0.01, shift_max=10):
    x_aug = x + np.random.normal(0, noise_level, size=x.shape)
    shift = np.random.randint(-shift_max, shift_max)
    x_aug = np.roll(x_aug, shift)
    return x_aug

# Callbacks personnalisés

def get_callbacks(patience=10, reduce_lr_patience=5, log_dir=None):
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', patience=reduce_lr_patience, factor=0.5, min_lr=1e-6)
    ]
    if log_dir:
        cb.append(callbacks.TensorBoard(log_dir=log_dir))
    return cb

# Visualisation des activations

def plot_activations(model, x, layer_names=None):
    intermediate_layer_model = models.Model(inputs=model.input, outputs=[l.output for l in model.layers if (layer_names is None or l.name in layer_names)])
    activations = intermediate_layer_model.predict(np.expand_dims(x, axis=0))
    for i, act in enumerate(activations):
        plt.figure(figsize=(10,2))
        plt.plot(act[0] if act.ndim==3 else act.flatten())
        plt.title(f'Activation {i} ({intermediate_layer_model.output_names[i]})')
        plt.show()

# CNN 1D pour signaux temporels
class CNN1DClassifier:
    def __init__(self, input_shape, n_classes, pretrained_weights=None):
        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(32, 7, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 5, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(n_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if pretrained_weights:
            self.model.load_weights(pretrained_weights)

    def fit(self, X, y, validation_data=None, epochs=50, batch_size=32, callbacks=None):
        return self.model.fit(X, y, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def save(self, path):
        self.model.save(path)

    @staticmethod
    def load(path):
        clf = CNN1DClassifier((1,1), 1)
        clf.model = models.load_model(path)
        return clf

# CNN 2D pour spectrogrammes
class CNN2DClassifier:
    def __init__(self, input_shape, n_classes, pretrained_weights=None):
        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(n_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if pretrained_weights:
            self.model.load_weights(pretrained_weights)

    def fit(self, X, y, validation_data=None, epochs=50, batch_size=32, callbacks=None):
        return self.model.fit(X, y, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def save(self, path):
        self.model.save(path)

    @staticmethod
    def load(path):
        clf = CNN2DClassifier((1,1,1), 1)
        clf.model = models.load_model(path)
        return clf

# LSTM pour séquences temporelles
class LSTMClassifier:
    def __init__(self, input_shape, n_classes, pretrained_weights=None):
        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(32),
            layers.Dense(64, activation='relu'),
            layers.Dense(n_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if pretrained_weights:
            self.model.load_weights(pretrained_weights)

    def fit(self, X, y, validation_data=None, epochs=50, batch_size=32, callbacks=None):
        return self.model.fit(X, y, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def save(self, path):
        self.model.save(path)

    @staticmethod
    def load(path):
        clf = LSTMClassifier((1,1), 1)
        clf.model = models.load_model(path)
        return clf

# Autoencoder pour détection d'anomalies
class AutoencoderAnomalyDetector:
    def __init__(self, input_shape, encoding_dim=32, pretrained_weights=None):
        input_layer = layers.Input(shape=input_shape)
        encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
        decoded = layers.Dense(input_shape[0], activation='linear')(encoded)
        self.model = models.Model(inputs=input_layer, outputs=decoded)
        self.model.compile(optimizer='adam', loss='mse')
        if pretrained_weights:
            self.model.load_weights(pretrained_weights)

    def fit(self, X, epochs=50, batch_size=32, validation_data=None, callbacks=None):
        return self.model.fit(X, X, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks=callbacks)

    def anomaly_score(self, X):
        recon = self.model.predict(X)
        return np.mean((X - recon)**2, axis=1)

    def save(self, path):
        self.model.save(path)

    @staticmethod
    def load(path):
        clf = AutoencoderAnomalyDetector((1,))
        clf.model = models.load_model(path)
        return clf

# Architecture hybride CNN-LSTM
class CNNLSTMClassifier:
    def __init__(self, input_shape, n_classes, pretrained_weights=None):
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Conv1D(32, 7, activation='relu', padding='same'))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Conv1D(64, 5, activation='relu', padding='same'))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.LSTM(64))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(n_classes, activation='softmax'))
        self.model = model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if pretrained_weights:
            self.model.load_weights(pretrained_weights)

    def fit(self, X, y, validation_data=None, epochs=50, batch_size=32, callbacks=None):
        return self.model.fit(X, y, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def save(self, path):
        self.model.save(path)

    @staticmethod
    def load(path):
        clf = CNNLSTMClassifier((1,1), 1)
        clf.model = models.load_model(path)
        return clf

"""
Exemple d'utilisation :

from src.deep_models import CNN1DClassifier, CNN2DClassifier, LSTMClassifier, AutoencoderAnomalyDetector, CNNLSTMClassifier, augment_signal, get_callbacks, plot_activations
import numpy as np

# Données fictives
X = np.random.randn(100, 1024, 1)
y = np.random.randint(0, 3, 100)

clf = CNN1DClassifier(input_shape=(1024,1), n_classes=3)
cb = get_callbacks()
clf.fit(X, y, epochs=5, callbacks=cb)

# Visualisation des activations
plot_activations(clf.model, X[0])

# Autoencoder pour détection d'anomalies
ae = AutoencoderAnomalyDetector(input_shape=(1024,))
ae.fit(X.squeeze(), epochs=5)
scores = ae.anomaly_score(X.squeeze())
""" 