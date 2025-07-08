# -*- coding: utf-8 -*-
"""
Pipeline complet de diagnostic moteur par IA

Ce script présente un pipeline de bout en bout, de la donnée brute à la prédiction, avec :
- Comparaison de tous les modèles implémentés
- Analyse des performances par type de défaillance
- Visualisation des résultats et métriques détaillées
- Exemples de prédictions sur nouvelles données
- Génération automatique de rapports avec LLM
- Intégration avec données de télémétrie simulées
- Recommandations pour le déploiement en production

Chaque étape est expliquée de façon pédagogique pour une présentation de stage.
"""

# =============================
# 1. Imports et configuration
# =============================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.classical_models import get_classical_models
from src.deep_models import get_deep_models
from src.feature_extraction import extract_features
from src.data_loader import load_data
from src.llm_integration import generate_report_llm

# =============================
# 2. Chargement et préparation des données
# =============================
print("\n## Chargement et préparation des données")
print("Nous chargeons les données brutes, extrayons les caractéristiques et préparons les jeux d'entraînement/test.\n")

# Chargement (adapter selon vos fonctions)
data = load_data(dataset='CWRU')  # ou 'MaFaulDa'
X, y = extract_features(data)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Taille train: {X_train.shape}, test: {X_test.shape}")

# =============================
# 3. Entraînement et comparaison des modèles
# =============================
print("\n## Entraînement et comparaison des modèles")
print("Nous comparons plusieurs modèles classiques et profonds sur les mêmes données.\n")

models = {}
results = {}

# Modèles classiques
for name, model in get_classical_models().items():
    print(f"\n--- Entraînement modèle classique : {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    models[name] = model
    results[name] = report

# Modèles profonds (exemple, à adapter selon vos fonctions)
for name, model in get_deep_models().items():
    print(f"\n--- Entraînement modèle profond : {name} ---")
    model.fit(X_train, y_train, epochs=5, verbose=1)  # epochs à ajuster
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    report = classification_report(y_test, y_pred, output_dict=True)
    models[name] = model
    results[name] = report

# =============================
# 4. Analyse des performances par type de défaillance
# =============================
print("\n## Analyse des performances par type de défaillance")

for name, report in results.items():
    print(f"\n{name} :")
    for label in report:
        if label.isdigit():
            print(f"  Classe {label} : Précision={report[label]['precision']:.2f}, Rappel={report[label]['recall']:.2f}, F1={report[label]['f1-score']:.2f}")

# =============================
# 5. Visualisation des résultats
# =============================
print("\n## Visualisation des résultats")

# Exemple : matrice de confusion pour le meilleur modèle
best_model = max(results, key=lambda k: results[k]['accuracy'])
y_pred = models[best_model].predict(X_test)
if y_pred.ndim > 1:
    y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred)
fig = px.imshow(cm, text_auto=True, title=f"Matrice de confusion - {best_model}")
fig.show()

# Courbes de scores par classe
scores_df = pd.DataFrame({name: {k: v['f1-score'] for k, v in report.items() if k.isdigit()} for name, report in results.items()})
scores_df.plot(kind='bar', figsize=(10,5))
plt.title('F1-score par classe et par modèle')
plt.ylabel('F1-score')
plt.xlabel('Classe')
plt.show()

# =============================
# 6. Exemples de prédictions sur nouvelles données
# =============================
print("\n## Exemples de prédictions sur nouvelles données")

# Simulation d'une nouvelle donnée (à adapter)
new_data = X_test.iloc[:5]
preds = models[best_model].predict(new_data)
if preds.ndim > 1:
    preds = np.argmax(preds, axis=1)
print("Prédictions :", preds)

# =============================
# 7. Génération automatique de rapports avec LLM
# =============================
print("\n## Génération automatique de rapports avec LLM")

# Exemple d'appel à un LLM pour générer un rapport
rapport = generate_report_llm(results, best_model=best_model)
print(rapport)

# =============================
# 8. Intégration avec données de télémétrie simulées
# =============================
print("\n## Intégration avec données de télémétrie simulées")

# Simulation de flux de données (exemple)
def simulate_telemetry(n=10):
    # Génère n signaux simulés (à adapter selon vos besoins)
    return X_test.sample(n)

telemetry_data = simulate_telemetry()
telemetry_preds = models[best_model].predict(telemetry_data)
if telemetry_preds.ndim > 1:
    telemetry_preds = np.argmax(telemetry_preds, axis=1)
print("Prédictions télémétrie :", telemetry_preds)

# =============================
# 9. Recommandations pour le déploiement en production
# =============================
print("\n## Recommandations pour le déploiement en production")
print("""
- Mettre en place une pipeline automatisée (prétraitement, prédiction, monitoring)
- Gérer la dérive de données (retraining régulier)
- Intégrer des alertes en cas de détection de défaut
- Documenter et valider les modèles sur des données réelles
- Prévoir une interface utilisateur pour l'exploitation terrain
- Sécuriser l'accès aux données et aux modèles

> Ces recommandations assurent robustesse, maintenabilité et valeur ajoutée en contexte industriel.
""") 