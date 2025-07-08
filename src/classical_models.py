import numpy as np
from sklearn.ensemble import RandomForestClassifier as SkRandomForest, GradientBoostingClassifier as SkGBC, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Configurations par défaut inspirées de la littérature sur le diagnostic de moteurs
DEFAULT_RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 12,
    'min_samples_split': 4,
    'class_weight': 'balanced_subsample',
    'random_state': 42
}
DEFAULT_SVM_PARAMS = {
    'C': 10,
    'kernel': 'rbf',
    'gamma': 'scale',
    'class_weight': 'balanced',
    'probability': True,
    'random_state': 42
}
DEFAULT_GBC_PARAMS = {
    'n_estimators': 150,
    'learning_rate': 0.08,
    'max_depth': 5,
    'subsample': 0.8,
    'validation_fraction': 0.15,
    'n_iter_no_change': 10,
    'random_state': 42
}

class RandomForestClassifier:
    def __init__(self, **kwargs):
        params = DEFAULT_RF_PARAMS.copy()
        params.update(kwargs)
        self.model = SkRandomForest(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self.model, path)

    @staticmethod
    def load(path):
        clf = RandomForestClassifier()
        clf.model = joblib.load(path)
        return clf

class SVMClassifier:
    def __init__(self, kernel='rbf', **kwargs):
        params = DEFAULT_SVM_PARAMS.copy()
        params.update(kwargs)
        params['kernel'] = kernel
        self.model = SVC(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError('SVM probability estimation not enabled.')

    def save(self, path):
        joblib.dump(self.model, path)

    @staticmethod
    def load(path):
        clf = SVMClassifier()
        clf.model = joblib.load(path)
        return clf

class GradientBoostingClassifier:
    def __init__(self, **kwargs):
        params = DEFAULT_GBC_PARAMS.copy()
        params.update(kwargs)
        self.model = SkGBC(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self.model, path)

    @staticmethod
    def load(path):
        clf = GradientBoostingClassifier()
        clf.model = joblib.load(path)
        return clf

class EnsembleClassifier:
    def __init__(self, estimators=None, voting='soft'):
        if estimators is None:
            estimators = [
                ('rf', SkRandomForest(**DEFAULT_RF_PARAMS)),
                ('svm', SVC(**DEFAULT_SVM_PARAMS)),
                ('gbc', SkGBC(**DEFAULT_GBC_PARAMS)),
            ]
        self.model = VotingClassifier(estimators=estimators, voting=voting)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self.model, path)

    @staticmethod
    def load(path):
        clf = EnsembleClassifier()
        clf.model = joblib.load(path)
        return clf

# Optimisation d'hyperparamètres avec GridSearchCV

def optimize_hyperparameters(model, param_grid, X, y, scoring='accuracy', cv=5, n_jobs=-1):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid = GridSearchCV(model, param_grid, scoring=scoring, cv=skf, n_jobs=n_jobs)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

# Validation croisée stratifiée et métriques spécialisées

def stratified_cross_val(model, X, y, cv=5, scoring='accuracy'):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
    return scores

def evaluate_model(model, X, y, class_names=None):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y, y_pred)
    # Précision par classe de défaillance
    per_class_acc = {k: v['precision'] for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']}
    return {
        'accuracy': acc,
        'per_class_precision': per_class_acc,
        'classification_report': report,
        'confusion_matrix': cm
    }

"""
Exemple d'utilisation :

from src.classical_models import RandomForestClassifier, SVMClassifier, GradientBoostingClassifier, EnsembleClassifier, optimize_hyperparameters, stratified_cross_val, evaluate_model

# X, y = ... (features et labels)
rf = RandomForestClassifier()
rf.fit(X, y)
print(evaluate_model(rf, X, y))

# Optimisation SVM
param_grid = {'C': [1, 10], 'kernel': ['rbf', 'poly'], 'gamma': ['scale', 'auto']}
best_svm, params, score = optimize_hyperparameters(SVC(), param_grid, X, y)

# Validation croisée
scores = stratified_cross_val(rf, X, y)
print('CV accuracy:', scores)

# Sauvegarde
rf.save('results/rf_model.joblib')
rf2 = RandomForestClassifier.load('results/rf_model.joblib')
""" 