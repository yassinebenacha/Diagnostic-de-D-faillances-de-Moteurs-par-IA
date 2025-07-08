# -*- coding: utf-8 -*-
"""
Exploration des données CWRU et MaFaulDa

Ce script propose une exploration détaillée des jeux de données CWRU (Case Western Reserve University) et MaFaulDa, utilisés pour le diagnostic de défaillances de moteurs électriques.

Nous allons :
- Charger et visualiser les signaux bruts
- Réaliser des analyses statistiques par type de défaillance
- Explorer les signatures temps-fréquence (spectrogrammes, transformées)
- Identifier les signatures caractéristiques
- Étudier les corrélations entre caractéristiques
- Utiliser des visualisations interactives (Plotly)
- Analyser la qualité des données
- Proposer des recommandations de prétraitement

Chaque section inclut des explications détaillées et des interprétations physiques.
"""

# =============================
# 1. Imports et configuration
# =============================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import signal
from scipy.stats import kurtosis, skew
from glob import glob

plt.style.use('seaborn-darkgrid')

# Répertoires des données (à adapter si besoin)
CWRU_PATH = '../data/raw/CWRU/'
MAFAULDA_PATH = '../data/raw/MaFaulDa/'

# =============================
# 2. Chargement des jeux de données
# =============================
print("\n## Chargement des jeux de données")
print("- CWRU : Données vibratoires de roulements, classées par type de défaut (extérieur, intérieur, bille, sain).\n- MaFaulDa : Données multi-capteurs (vibration, courant, etc.) pour différents scénarios de défaillance.\n")
print("> Physique : Les signaux vibratoires et électriques sont sensibles aux défauts mécaniques (p.ex. roulements, déséquilibres) et électriques (p.ex. court-circuit).\n")

def load_CWRU_files(path):
    files = glob(os.path.join(path, '*.csv'))
    data = {}
    for f in files:
        label = os.path.basename(f).split('_')[0]
        df = pd.read_csv(f)
        data[label] = df
    return data

def load_MaFaulDa_files(path):
    files = glob(os.path.join(path, '*.csv'))
    data = {}
    for f in files:
        label = os.path.basename(f).split('_')[0]
        df = pd.read_csv(f)
        data[label] = df
    return data

cwru_data = load_CWRU_files(CWRU_PATH)
mafaulda_data = load_MaFaulDa_files(MAFAULDA_PATH)

print(f'CWRU: {list(cwru_data.keys())}')
print(f'MaFaulDa: {list(mafaulda_data.keys())}')

# =============================
# 3. Visualisation des signaux bruts
# =============================
print("\n## Visualisation des signaux bruts")
print("> Interprétation physique : Les défauts modifient la forme d'onde (pics, modulations, bruit).\n")

def plot_signal(df, column='signal', title='Signal', n=2000):
    plt.figure(figsize=(12,4))
    plt.plot(df[column].values[:n])
    plt.title(title)
    plt.xlabel('Echantillon')
    plt.ylabel('Amplitude')
    plt.show()

# Exemple : signal sain et défaillant
for label, df in list(cwru_data.items())[:2]:
    plot_signal(df, title=f'CWRU - {label}')

# =============================
# 4. Analyse statistique par type de défaillance
# =============================
print("\n## Analyse statistique par type de défaillance")
print("> Physique : Les défauts augmentent souvent la variance, la kurtosis (pics) et la skewness (asymétrie).\n")

def compute_stats(df, column='signal'):
    x = df[column].values
    return {
        'mean': np.mean(x),
        'std': np.std(x),
        'kurtosis': kurtosis(x),
        'skewness': skew(x)
    }

stats_cwru = {label: compute_stats(df) for label, df in cwru_data.items()}
stats_df = pd.DataFrame(stats_cwru).T
print(stats_df)

# =============================
# 5. Visualisation temps-fréquence (spectrogrammes, transformées)
# =============================
print("\n## Visualisation temps-fréquence (spectrogrammes, transformées)")
print("> Physique : Les défauts introduisent des composantes fréquentielles spécifiques (harmoniques, bandes latérales).\n")

def plot_fft(df, column='signal', fs=12000, title='FFT'):
    x = df[column].values
    f, Pxx = signal.welch(x, fs=fs, nperseg=1024)
    plt.figure(figsize=(10,4))
    plt.semilogy(f, Pxx)
    plt.title(title)
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('PSD')
    plt.show()

for label, df in list(cwru_data.items())[:2]:
    plot_fft(df, title=f'CWRU FFT - {label}')

def plot_spectrogram(df, column='signal', fs=12000, title='Spectrogramme'):
    x = df[column].values
    f, t, Sxx = signal.spectrogram(x, fs=fs)
    plt.figure(figsize=(10,4))
    plt.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
    plt.ylabel('Fréquence [Hz]')
    plt.xlabel('Temps [s]')
    plt.title(title)
    plt.colorbar(label='dB')
    plt.show()

for label, df in list(cwru_data.items())[:2]:
    plot_spectrogram(df, title=f'CWRU Spectrogramme - {label}')

# =============================
# 6. Identification des signatures caractéristiques
# =============================
print("\n## Identification des signatures caractéristiques")
print("> Physique : Les défauts de roulement génèrent des fréquences de battement spécifiques (BPFO, BPFI, BSF, FTF).\n")

def find_peaks_fft(df, column='signal', fs=12000, n_peaks=5):
    x = df[column].values
    f, Pxx = signal.welch(x, fs=fs, nperseg=1024)
    peaks, _ = signal.find_peaks(Pxx, height=np.max(Pxx)*0.1)
    peak_freqs = f[peaks][:n_peaks]
    return peak_freqs

for label, df in list(cwru_data.items())[:2]:
    peaks = find_peaks_fft(df)
    print(f'{label}: pics principaux à {peaks} Hz')

# =============================
# 7. Corrélations entre caractéristiques
# =============================
print("\n## Corrélations entre caractéristiques")
print("> Physique : Certaines caractéristiques sont redondantes, d'autres complémentaires pour le diagnostic.\n")

features = []
for label, df in cwru_data.items():
    stats = compute_stats(df)
    stats['label'] = label
    features.append(stats)
features_df = pd.DataFrame(features)

corr = features_df.drop('label', axis=1).corr()
fig = px.imshow(corr, text_auto=True, title='Corrélation entre caractéristiques')
fig.show()

# =============================
# 8. Visualisations interactives (Plotly)
# =============================
print("\n## Visualisations interactives (Plotly)")
print("> Intérêt : Permet d'identifier visuellement des clusters ou anomalies.\n")

fig = px.scatter(features_df, x='kurtosis', y='std', color='label', title='Kurtosis vs Std par type de défaut')
fig.show()

# =============================
# 9. Analyse de la qualité des données
# =============================
print("\n## Analyse de la qualité des données")
print("> Physique : Une mauvaise qualité de données peut masquer ou imiter des défauts.\n")

# Valeurs manquantes
for label, df in cwru_data.items():
    print(f'{label}: {df.isnull().sum().sum()} valeurs manquantes')

# Outliers (exemple simple)
for label, df in cwru_data.items():
    x = df['signal'].values
    outliers = np.sum(np.abs(x - np.mean(x)) > 5*np.std(x))
    print(f'{label}: {outliers} outliers (>5σ)')

# =============================
# 10. Recommandations pour le prétraitement
# =============================
print("\n## Recommandations pour le prétraitement")
print("""
Sur la base de l'analyse précédente, nous recommandons :
- Filtrage : Suppression du bruit haute fréquence et des artefacts basse fréquence.
- Normalisation : Pour comparer les signaux entre classes.
- Suppression des outliers : Pour éviter les biais d'apprentissage.
- Extraction de fenêtres : Pour traiter des signaux longs.
- Augmentation de données : Si déséquilibre de classes.

> Physique : Un bon prétraitement améliore la détection des signatures de défauts et la robustesse des modèles.
""") 