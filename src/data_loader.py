import os
import logging
import numpy as np
import pandas as pd
from scipy.io import loadmat
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class DataLoaderError(Exception):
    """Exception personnalisée pour les erreurs de chargement de données."""
    pass

class GenericDataLoader:
    """
    Classe générique pour charger des données de télémétrie personnalisées.
    Supporte CSV, MAT, HDF5. Gère la normalisation, l'échantillonnage et le split train/test.
    """
    def __init__(self, filepath, label_col=None, meta_cols=None, file_format=None):
        self.filepath = filepath
        self.label_col = label_col
        self.meta_cols = meta_cols or []
        self.file_format = file_format or self._infer_format()
        self.data = None
        self.labels = None
        self.meta = None
        logger.info(f"Initialisation GenericDataLoader pour {filepath} (format: {self.file_format})")

    def _infer_format(self):
        ext = os.path.splitext(self.filepath)[-1].lower()
        if ext == '.csv':
            return 'csv'
        elif ext == '.mat':
            return 'mat'
        elif ext in ['.h5', '.hdf5']:
            return 'hdf5'
        else:
            raise DataLoaderError(f"Format de fichier non supporté: {ext}")

    def load(self):
        try:
            if self.file_format == 'csv':
                df = pd.read_csv(self.filepath)
                self.data = df.drop(self.meta_cols + ([self.label_col] if self.label_col else []), axis=1)
                self.labels = df[self.label_col] if self.label_col else None
                self.meta = df[self.meta_cols] if self.meta_cols else None
            elif self.file_format == 'mat':
                mat = loadmat(self.filepath)
                # Hypothèse: données dans 'X', labels dans 'y', métadonnées dans 'meta' (adapter si besoin)
                self.data = pd.DataFrame(mat.get('X', []))
                self.labels = pd.Series(mat.get('y', []).flatten()) if 'y' in mat else None
                self.meta = pd.DataFrame(mat.get('meta', [])) if 'meta' in mat else None
            elif self.file_format == 'hdf5':
                with h5py.File(self.filepath, 'r') as f:
                    # Hypothèse: datasets 'X', 'y', 'meta' (adapter si besoin)
                    self.data = pd.DataFrame(f['X'][:]) if 'X' in f else None
                    self.labels = pd.Series(f['y'][:]) if 'y' in f else None
                    self.meta = pd.DataFrame(f['meta'][:]) if 'meta' in f else None
            else:
                raise DataLoaderError(f"Format de fichier non supporté: {self.file_format}")
            logger.info(f"Chargement réussi: {self.filepath}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            raise DataLoaderError(e)
        return self

    def normalize(self, method='standard'):
        if self.data is None:
            raise DataLoaderError("Aucune donnée à normaliser. Chargez d'abord les données.")
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Méthode de normalisation inconnue.")
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)
        logger.info(f"Normalisation appliquée: {method}")
        return self

    def sample(self, n=None, frac=None, random_state=42):
        if self.data is None:
            raise DataLoaderError("Aucune donnée à échantillonner.")
        if n:
            idx = self.data.sample(n=n, random_state=random_state).index
        elif frac:
            idx = self.data.sample(frac=frac, random_state=random_state).index
        else:
            return self
        self.data = self.data.loc[idx]
        if self.labels is not None:
            self.labels = self.labels.loc[idx]
        if self.meta is not None:
            self.meta = self.meta.loc[idx]
        logger.info(f"Échantillonnage: {n or frac}")
        return self

    def train_test_split(self, test_size=0.2, random_state=42, stratify=True):
        if self.data is None:
            raise DataLoaderError("Aucune donnée à diviser.")
        strat = self.labels if stratify and self.labels is not None else None
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=random_state, stratify=strat
        )
        logger.info(f"Split train/test: {1-test_size:.0%} / {test_size:.0%}")
        return X_train, X_test, y_train, y_test

class CWRUDataLoader(GenericDataLoader):
    """
    Loader spécialisé pour le dataset Case Western Reserve University (CWRU).
    Gère la structure spécifique, les labels de défaillance et les métadonnées.
    """
    def __init__(self, filepath, label_map=None):
        super().__init__(filepath, label_col='label', meta_cols=['rpm', 'load'], file_format=None)
        self.label_map = label_map or {}
        logger.info("Initialisation CWRUDataLoader")

    def load(self):
        super().load()
        # Adapter ici selon la structure réelle du dataset CWRU
        if self.labels is not None and self.label_map:
            self.labels = self.labels.map(self.label_map)
        logger.info("Chargement CWRU terminé")
        return self

class MaFaulDaDataLoader(GenericDataLoader):
    """
    Loader spécialisé pour le dataset MaFaulDa.
    Gère la structure spécifique, les labels de défaillance et les métadonnées.
    """
    def __init__(self, filepath, label_map=None):
        super().__init__(filepath, label_col='fault_type', meta_cols=['rpm', 'load'], file_format=None)
        self.label_map = label_map or {}
        logger.info("Initialisation MaFaulDaDataLoader")

    def load(self):
        super().load()
        # Adapter ici selon la structure réelle du dataset MaFaulDa
        if self.labels is not None and self.label_map:
            self.labels = self.labels.map(self.label_map)
        logger.info("Chargement MaFaulDa terminé")
        return self

"""
Exemples d'utilisation :

# Pour un CSV générique
loader = GenericDataLoader('data/datasets/mesures.csv', label_col='fault', meta_cols=['rpm', 'temp'])
loader.load().normalize('minmax').sample(frac=0.5)
X_train, X_test, y_train, y_test = loader.train_test_split()

# Pour CWRU
cwru_loader = CWRUDataLoader('data/datasets/cwru_data.csv', label_map={0: 'healthy', 1: 'faulty'})
cwru_loader.load().normalize()
X_train, X_test, y_train, y_test = cwru_loader.train_test_split()

# Pour MaFaulDa
mafaulda_loader = MaFaulDaDataLoader('data/datasets/mafaulda.mat', label_map={0: 'normal', 1: 'fault'})
mafaulda_loader.load().normalize('minmax')
X_train, X_test, y_train, y_test = mafaulda_loader.train_test_split()
""" 