#!/bin/bash

# Script d'installation automatique pour le projet Diagnostic de Défaillances de Moteurs par IA

if command -v conda &> /dev/null; then
    echo "Conda détecté. Création de l'environnement via environment.yml..."
    conda env create -f environment.yml || conda env update -f environment.yml
    echo "Activez l'environnement avec : conda activate diagnostic-moteur-ia"
else
    echo "Conda non détecté. Installation via pip et requirements.txt..."
    python -m venv venv
    source venv/bin/activate || source venv/Scripts/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "Activez l'environnement avec : source venv/bin/activate (Linux/Mac) ou venv\\Scripts\\activate (Windows)"
fi 