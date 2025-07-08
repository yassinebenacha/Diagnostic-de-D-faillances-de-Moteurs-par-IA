import numpy as np
import scipy.signal as signal
import pywt
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

class SignalProcessingError(Exception):
    """Exception personnalisée pour les erreurs de traitement du signal."""
    pass

def lowpass_filter(x, fs, cutoff, order=4, plot=False):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, x)
    if plot:
        plt.figure()
        plt.plot(x, label='Signal brut')
        plt.plot(y, label='Filtré passe-bas')
        plt.legend(); plt.title('Filtrage passe-bas')
        plt.show()
    return y

def highpass_filter(x, fs, cutoff, order=4, plot=False):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, x)
    if plot:
        plt.figure()
        plt.plot(x, label='Signal brut')
        plt.plot(y, label='Filtré passe-haut')
        plt.legend(); plt.title('Filtrage passe-haut')
        plt.show()
    return y

def bandpass_filter(x, fs, lowcut, highcut, order=4, plot=False):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, x)
    if plot:
        plt.figure()
        plt.plot(x, label='Signal brut')
        plt.plot(y, label='Filtré passe-bande')
        plt.legend(); plt.title('Filtrage passe-bande')
        plt.show()
    return y

def compute_fft(x, fs, plot=False):
    N = len(x)
    f = np.fft.rfftfreq(N, 1/fs)
    X = np.abs(np.fft.rfft(x))
    if plot:
        plt.figure()
        plt.plot(f, X)
        plt.title('FFT')
        plt.xlabel('Fréquence (Hz)')
        plt.ylabel('Amplitude')
        plt.show()
    return f, X

def compute_spectrogram(x, fs, nperseg=256, plot=False):
    f, t, Sxx = signal.spectrogram(x, fs, nperseg=nperseg)
    if plot:
        plt.figure()
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.ylabel('Fréquence [Hz]')
        plt.xlabel('Temps [s]')
        plt.title('Spectrogramme')
        plt.colorbar(label='dB')
        plt.show()
    return f, t, Sxx

def extract_time_features(x):
    features = {
        'rms': np.sqrt(np.mean(x**2)),
        'kurtosis': kurtosis(x),
        'skewness': skew(x),
        'mean': np.mean(x),
        'std': np.std(x),
        'max': np.max(x),
        'min': np.min(x)
    }
    return features

def extract_freq_features(x, fs, n_peaks=3):
    f, X = compute_fft(x, fs)
    centroid = np.sum(f * X) / np.sum(X)
    peaks, _ = signal.find_peaks(X, height=np.max(X)*0.1)
    peak_freqs = f[peaks][:n_peaks]
    peak_amps = X[peaks][:n_peaks]
    features = {
        'spectral_centroid': centroid,
        'peak_freqs': peak_freqs,
        'peak_amps': peak_amps
    }
    return features

def envelope_analysis(x, fs, plot=False):
    analytic_signal = signal.hilbert(x)
    envelope = np.abs(analytic_signal)
    if plot:
        plt.figure()
        plt.plot(x, label='Signal')
        plt.plot(envelope, label='Enveloppe')
        plt.legend(); plt.title('Analyse d\'enveloppe')
        plt.show()
    return envelope

def wavelet_transform(x, wavelet='db4', level=4, plot=False):
    coeffs = pywt.wavedec(x, wavelet, level=level)
    if plot:
        plt.figure(figsize=(10, 2*len(coeffs)))
        for i, c in enumerate(coeffs):
            plt.subplot(len(coeffs), 1, i+1)
            plt.plot(c)
            plt.title(f'Coefficients niveau {i}')
        plt.tight_layout()
        plt.show()
    return coeffs

def detect_peaks(x, height=None, distance=None, plot=False):
    peaks, properties = signal.find_peaks(x, height=height, distance=distance)
    if plot:
        plt.figure()
        plt.plot(x)
        plt.plot(peaks, x[peaks], 'rx')
        plt.title('Détection de pics')
        plt.show()
    return peaks, properties

def detect_harmonics(x, fs, fundamental_freq, n_harmonics=5, tol=2):
    f, X = compute_fft(x, fs)
    harmonics = []
    for n in range(1, n_harmonics+1):
        target = n * fundamental_freq
        idx = np.where((f > target-tol) & (f < target+tol))[0]
        if len(idx) > 0:
            harmonics.append((f[idx[0]], X[idx[0]]))
    return harmonics

class SignalProcessor:
    """
    Classe de traitement du signal avec méthodes chainables pour pipeline.
    """
    def __init__(self, x, fs):
        self.x = np.asarray(x)
        self.fs = fs
        self.history = []

    def lowpass(self, cutoff, order=4, plot=False):
        self.x = lowpass_filter(self.x, self.fs, cutoff, order, plot)
        self.history.append(('lowpass', cutoff))
        return self

    def highpass(self, cutoff, order=4, plot=False):
        self.x = highpass_filter(self.x, self.fs, cutoff, order, plot)
        self.history.append(('highpass', cutoff))
        return self

    def bandpass(self, lowcut, highcut, order=4, plot=False):
        self.x = bandpass_filter(self.x, self.fs, lowcut, highcut, order, plot)
        self.history.append(('bandpass', (lowcut, highcut)))
        return self

    def fft(self, plot=False):
        self.fft_freqs, self.fft_amps = compute_fft(self.x, self.fs, plot)
        return self

    def spectrogram(self, nperseg=256, plot=False):
        self.spec_f, self.spec_t, self.spec_Sxx = compute_spectrogram(self.x, self.fs, nperseg, plot)
        return self

    def envelope(self, plot=False):
        self.envelope = envelope_analysis(self.x, self.fs, plot)
        return self

    def wavelet(self, wavelet='db4', level=4, plot=False):
        self.wavelet_coeffs = wavelet_transform(self.x, wavelet, level, plot)
        return self

    def time_features(self):
        self.time_feats = extract_time_features(self.x)
        return self

    def freq_features(self, n_peaks=3):
        self.freq_feats = extract_freq_features(self.x, self.fs, n_peaks)
        return self

    def detect_peaks(self, height=None, distance=None, plot=False):
        self.peaks, self.peak_props = detect_peaks(self.x, height, distance, plot)
        return self

    def detect_harmonics(self, fundamental_freq, n_harmonics=5, tol=2):
        self.harmonics = detect_harmonics(self.x, self.fs, fundamental_freq, n_harmonics, tol)
        return self

    def get(self):
        return self.x

"""
Exemple d'utilisation :

import numpy as np
from src.signal_processing import SignalProcessor

fs = 12000
x = np.sin(2*np.pi*50*np.arange(0,1,1/fs)) + 0.5*np.random.randn(fs)

sp = SignalProcessor(x, fs)
sp.lowpass(500, plot=True).highpass(10, plot=True).fft(plot=True)
sp.spectrogram(plot=True)
sp.envelope(plot=True)
sp.wavelet(plot=True)
print(sp.time_features().time_feats)
print(sp.freq_features().freq_feats)
sp.detect_peaks(plot=True)
sp.detect_harmonics(fundamental_freq=50)
""" 