import numpy as np
import pandas as pd
import joblib
import logging
from src.feature_extraction import MotorFeatureExtractor
from src.classical_models import RandomForestClassifier, SVMClassifier, GradientBoostingClassifier, EnsembleClassifier
from src.evaluation import multiclass_metrics
from typing import Any, Dict

# Logging configuration
logger = logging.getLogger('MotorFaultPredictor')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('results/predictions.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

class MotorFaultPredictor:
    """
    Pipeline de prédiction temps réel pour moteurs électriques.
    """
    def __init__(self, model_path, feature_extractor: MotorFeatureExtractor, class_names=None, drift_threshold=0.2):
        self.model = joblib.load(model_path)
        self.feature_extractor = feature_extractor
        self.class_names = class_names
        self.drift_threshold = drift_threshold
        self.last_feature_mean = None
        logger.info(f"Modèle chargé depuis {model_path}")

    def predict(self, raw_signal: np.ndarray) -> Dict[str, Any]:
        features = self.feature_extractor.extract_all(raw_signal)
        X = np.array([list(features.values())])
        y_pred = self.model.predict(X)[0]
        y_proba = self.model.predict_proba(X)[0]
        confidence = np.max(y_proba)
        drift = self.detect_drift(features)
        alert = confidence < 0.6 or drift
        result = {
            'prediction': self.class_names[y_pred] if self.class_names else y_pred,
            'confidence': confidence,
            'proba': y_proba.tolist(),
            'drift': drift,
            'alert': alert,
            'features': features
        }
        logger.info(f"Prediction: {result}")
        return result

    def detect_drift(self, features: Dict[str, float]) -> bool:
        # Simple drift: compare la moyenne des features à l'historique
        feat_mean = np.mean(list(features.values()))
        if self.last_feature_mean is None:
            self.last_feature_mean = feat_mean
            return False
        drift = abs(feat_mean - self.last_feature_mean) > self.drift_threshold * abs(self.last_feature_mean)
        self.last_feature_mean = feat_mean
        return drift

    def stream_predict(self, signal_stream):
        """Prédiction sur un flux de signaux (générateur ou liste)."""
        for raw_signal in signal_stream:
            yield self.predict(raw_signal)

# API REST avec FastAPI
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Motor Fault Predictor API")
predictor: MotorFaultPredictor = None

class PredictRequest(BaseModel):
    signal: list

@app.on_event("startup")
def load_predictor():
    global predictor
    # À adapter selon vos chemins et paramètres
    feature_extractor = MotorFeatureExtractor(fs=12000)
    predictor = MotorFaultPredictor('results/rf_model.joblib', feature_extractor, class_names=["Normal", "Défaillance"])

@app.post("/predict")
def predict(req: PredictRequest):
    signal = np.array(req.signal)
    result = predictor.predict(signal)
    return result

# Interface web Streamlit

def run_streamlit():
    import streamlit as st
    st.title("Diagnostic de Défaillances de Moteurs (IA)")
    uploaded = st.file_uploader("Charger un fichier de télémétrie (CSV)")
    if uploaded:
        df = pd.read_csv(uploaded)
        signal = df.iloc[:,0].values
        result = predictor.predict(signal)
        st.write("## Prédiction", result['prediction'])
        st.write("Confiance :", result['confidence'])
        st.write("Alerte :", result['alert'])
        st.write("Dérive détectée :", result['drift'])
        st.write("Features :", result['features'])

# Exemple d'intégration GYM Global USSU Management
"""
# Intégration GYM USSU (pseudocode)
from src.predictor import MotorFaultPredictor
# ...
for telemetry in gym_ussu_stream():
    result = predictor.predict(telemetry['signal'])
    if result['alert']:
        send_alert_to_gym_ussu(telemetry['machine_id'], result)
"""

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'api':
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif len(sys.argv) > 1 and sys.argv[1] == 'streamlit':
        run_streamlit()
    else:
        print("Usage: python src/predictor.py [api|streamlit]") 