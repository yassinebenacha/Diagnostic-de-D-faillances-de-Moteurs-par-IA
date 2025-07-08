import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from statsmodels.stats.contingency_tables import mcnemar
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.pyplot as plt
import os

# Métriques multi-classes

def multiclass_metrics(y_true, y_pred, average='macro'):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
        'report': classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    }

# Matrice de confusion interactive

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, title='Matrice de confusion'):
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    z = cm
    x = class_names
    y = class_names
    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues', showscale=True)
    fig.update_layout(title=title, xaxis_title='Prédit', yaxis_title='Réel')
    fig.show()
    return fig

# Courbes ROC et AUC pour chaque classe

def plot_multiclass_roc(y_true, y_score, n_classes, class_names=None):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_true_bin = pd.get_dummies(y_true, columns=range(n_classes)).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fig = go.Figure()
    for i in range(n_classes):
        fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], mode='lines', name=f'Classe {class_names[i] if class_names else i} (AUC={roc_auc[i]:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aléatoire', line=dict(dash='dash')))
    fig.update_layout(title='Courbes ROC multi-classes', xaxis_title='Faux positifs', yaxis_title='Vrais positifs')
    fig.show()
    return fig, roc_auc

# Analyse des erreurs par type de défaillance

def error_analysis(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    errors = {}
    for i, true_label in enumerate(cm):
        for j, count in enumerate(true_label):
            if i != j and count > 0:
                key = f'{class_names[i] if class_names else i}→{class_names[j] if class_names else j}'
                errors[key] = int(count)
    return errors

# Validation croisée temporelle

def temporal_cross_val(model, X, y, n_splits=5, scoring='accuracy'):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring)
    return scores

# Test de significativité (ex: t-test sur scores)

def significance_test(scores1, scores2):
    from scipy.stats import ttest_rel
    stat, p = ttest_rel(scores1, scores2)
    return {'t_stat': stat, 'p_value': p}

# Test de McNemar pour comparaison de modèles

def mcnemar_test(y_true, y_pred1, y_pred2):
    tb = confusion_matrix(y_pred1 == y_true, y_pred2 == y_true)
    result = mcnemar(tb, exact=True)
    return {'statistic': result.statistic, 'p_value': result.pvalue}

# Génération de rapports HTML/PDF

def generate_report(metrics, confusion_fig=None, roc_fig=None, errors=None, output_path='results/report.html', title='Rapport d\'évaluation'):
    html = f"<h1>{title}</h1>"
    html += "<h2>Métriques globales</h2><pre>" + str(metrics) + "</pre>"
    if confusion_fig:
        cm_html = pio.to_html(confusion_fig, full_html=False, include_plotlyjs='cdn')
        html += "<h2>Matrice de confusion</h2>" + cm_html
    if roc_fig:
        roc_html = pio.to_html(roc_fig, full_html=False, include_plotlyjs='cdn')
        html += "<h2>Courbes ROC</h2>" + roc_html
    if errors:
        html += "<h2>Analyse des erreurs</h2><pre>" + str(errors) + "</pre>"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    try:
        import pdfkit
        pdfkit.from_file(output_path, output_path.replace('.html', '.pdf'))
    except Exception:
        pass
    return output_path

"""
Exemple d'utilisation :

from src.evaluation import multiclass_metrics, plot_confusion_matrix, plot_multiclass_roc, error_analysis, temporal_cross_val, significance_test, mcnemar_test, generate_report

# y_true, y_pred, y_score = ...
metrics = multiclass_metrics(y_true, y_pred)
fig_cm = plot_confusion_matrix(y_true, y_pred, class_names)
fig_roc, aucs = plot_multiclass_roc(y_true, y_score, n_classes, class_names)
errors = error_analysis(y_true, y_pred, class_names)
scores = temporal_cross_val(model, X, y)
sig = significance_test(scores1, scores2)
mcnemar_res = mcnemar_test(y_true, y_pred1, y_pred2)
generate_report(metrics, fig_cm, fig_roc, errors)
""" 