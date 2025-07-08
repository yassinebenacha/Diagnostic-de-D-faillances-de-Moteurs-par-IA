import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from src.signal_processing import compute_fft

# Références sur les fréquences caractéristiques des roulements
# BPFI (Ball Pass Frequency Inner), BPFO (Ball Pass Frequency Outer), BSF (Ball Spin Frequency)
# Voir : https://www.machinedyn.com/calculators/bearing-defect-frequency.php

def bearing_frequencies(shaft_freq, nb_rollers, roller_diam, pitch_diam, contact_angle_deg):
    """
    Calcule les fréquences caractéristiques des roulements.
    Retourne un dict avec BPFI, BPFO, BSF, FTF.
    """
    ca = np.deg2rad(contact_angle_deg)
    BPFI = 0.5 * nb_rollers * shaft_freq * (1 + roller_diam/pitch_diam * np.cos(ca))
    BPFO = 0.5 * nb_rollers * shaft_freq * (1 - roller_diam/pitch_diam * np.cos(ca))
    BSF = (pitch_diam/(2*roller_diam)) * shaft_freq * (1 - (roller_diam/pitch_diam * np.cos(ca))**2)
    FTF = 0.5 * shaft_freq * (1 - roller_diam/pitch_diam * np.cos(ca))
    return {'BPFI': BPFI, 'BPFO': BPFO, 'BSF': BSF, 'FTF': FTF}

def statistical_features(x):
    return {
        'mean': np.mean(x),
        'std': np.std(x),
        'var': np.var(x),
        'skewness': skew(x),
        'kurtosis': kurtosis(x),
        'crest_factor': np.max(np.abs(x)) / np.sqrt(np.mean(x**2)),
        'shape_factor': np.sqrt(np.mean(x**2)) / np.mean(np.abs(x)),
    }

def spectral_features(x, fs, defect_freqs=None, nperseg=1024):
    f, Pxx = welch(x, fs, nperseg=nperseg)
    features = {}
    if defect_freqs:
        for name, freq in defect_freqs.items():
            idx = np.argmin(np.abs(f - freq))
            features[f'power_{name}'] = Pxx[idx]
    # Spectral entropy
    Pxx_norm = Pxx / np.sum(Pxx)
    spectral_entropy = -np.sum(Pxx_norm * np.log(Pxx_norm + 1e-12))
    features['spectral_entropy'] = spectral_entropy
    # Energy in bands (split spectrum in 4 bands)
    bands = np.array_split(Pxx, 4)
    for i, band in enumerate(bands):
        features[f'band_energy_{i+1}'] = np.sum(band)
    return features

def time_freq_features(x, fs, nperseg=256):
    from scipy.signal import spectrogram
    f, t, Sxx = spectrogram(x, fs, nperseg=nperseg)
    # Energy per band (4 bands)
    bands = np.array_split(Sxx, 4, axis=0)
    features = {}
    for i, band in enumerate(bands):
        features[f'tf_band_energy_{i+1}'] = np.sum(band)
    # Mean spectral entropy over time
    Sxx_norm = Sxx / (np.sum(Sxx, axis=0, keepdims=True) + 1e-12)
    entropy = -np.sum(Sxx_norm * np.log(Sxx_norm + 1e-12), axis=0)
    features['mean_tf_entropy'] = np.mean(entropy)
    return features

class MotorFeatureExtractor:
    """
    Extracteur de caractéristiques pour le diagnostic de moteurs électriques.
    Permet l'extraction statistique, fréquentielle, temps-fréquence, et la sélection/export.
    """
    def __init__(self, fs, shaft_freq=None, nb_rollers=None, roller_diam=None, pitch_diam=None, contact_angle_deg=0):
        self.fs = fs
        self.defect_freqs = None
        if all(v is not None for v in [shaft_freq, nb_rollers, roller_diam, pitch_diam]):
            self.defect_freqs = bearing_frequencies(shaft_freq, nb_rollers, roller_diam, pitch_diam, contact_angle_deg)

    def extract_statistical(self, x):
        return statistical_features(x)

    def extract_spectral(self, x):
        return spectral_features(x, self.fs, self.defect_freqs)

    def extract_time_freq(self, x):
        return time_freq_features(x, self.fs)

    def extract_all(self, x):
        feats = {}
        feats.update(self.extract_statistical(x))
        feats.update(self.extract_spectral(x))
        feats.update(self.extract_time_freq(x))
        return feats

    def feature_pipeline(self, X, select=None):
        """
        X : array-like (n_samples, n_points)
        select : liste de noms de caractéristiques à garder (ou None pour tout)
        Retourne un array (n_samples, n_features) et la liste des noms de features.
        """
        features_list = []
        for x in X:
            feats = self.extract_all(x)
            if select:
                feats = {k: v for k, v in feats.items() if k in select}
            features_list.append(feats)
        feature_names = list(features_list[0].keys())
        feature_matrix = np.array([[f[n] for n in feature_names] for f in features_list])
        return feature_matrix, feature_names

    def to_sklearn(self, X, select=None):
        """
        Exporte les features au format compatible scikit-learn (numpy array).
        """
        return self.feature_pipeline(X, select)[0]

"""
Exemple d'utilisation :

from src.feature_extraction import MotorFeatureExtractor
import numpy as np

fs = 12000
# Paramètres roulement fictifs
shaft_freq = 30  # Hz
nb_rollers = 8
roller_diam = 0.01  # m
pitch_diam = 0.04  # m
contact_angle = 15  # deg

extractor = MotorFeatureExtractor(fs, shaft_freq, nb_rollers, roller_diam, pitch_diam, contact_angle)

x = np.random.randn(fs)
features = extractor.extract_all(x)
print(features)

# Pour un ensemble de signaux
X = np.random.randn(10, fs)
X_feats, feat_names = extractor.feature_pipeline(X)
print(X_feats.shape, feat_names)
""" 