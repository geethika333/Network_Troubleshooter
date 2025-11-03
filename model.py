 # connectivity_model.py
# Minimal ML model that TRAINS FROM A CSV FILE and predicts a fault domain.

import argparse, json, os, sys
from typing import Dict, Any
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load

MODEL_PATH = "connectivity_model.joblib"

LABELS = ["END_DEVICE", "ACCESS_LINK", "LOCAL_NETWORK", "ISP", "UPSTREAM"]
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for l, i in L2I.items()}

NUMERIC = [
    "rtt_gw_ms","rtt_public_ms","loss_public_pct","dns_latency_ms",
    "tr_last_hop_idx","rssi_dbm","dhcp_lease_age_min",
    "jitter_ms","dns_ttl_avg","dns_response_ip_diversity"
]
CATEG = [
    "ping_gw_ok","ping_public_ok","ping_dns_ip_ok","dns_ok",
    "ip_valid","eth_link_up","wifi_connected","error_code"
]
ALL_FEATS = NUMERIC + CATEG

def load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in (ALL_FEATS+["label"]) if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    X = df[ALL_FEATS].copy()
    y = df["label"].map(L2I)
    if y.isna().any():
        bad = df.loc[y.isna(),"label"].unique().tolist()
        raise ValueError(f"Unknown labels in dataset: {bad}")
    # basic numeric cleanup
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].mean() if not X[c].dropna().empty else 0.0)
    return X, y.astype(int)

def build_pipeline(seed=0) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEG),
    ])
    clf = MLPClassifier(
        hidden_layer_sizes=(128,64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=300,
        random_state=seed
    )
    return Pipeline([("pre", pre), ("clf", clf)])

def train_and_save(data_path: str, model_path: str = MODEL_PATH):
    X, y = load_dataset(data_path)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=7)
    pipe = build_pipeline()
    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)
    acc = accuracy_score(yte, yhat)
    print(f"[train] test accuracy = {acc:.3f}")
    print(classification_report(yte, yhat, target_names=LABELS, digits=3))
    dump(pipe, model_path)
    print(f"[train] saved model -> {model_path}")

def predict_from_features(feats: Dict[str, Any], model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Train first.")
    pipe: Pipeline = load(model_path)
    # ensure all keys present
    for k in ALL_FEATS:
        feats.setdefault(k, 0.0 if k in NUMERIC else "no")
    X = pd.DataFrame([feats], columns=ALL_FEATS)
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    proba = pipe.predict_proba(X)[0]
    pred_id = int(np.argmax(proba))
    return {
        "label": I2L[pred_id],
        "confidence": float(proba[pred_id]),
        "proba_per_class": {I2L[i]: float(p) for i, p in enumerate(proba)}
    }

def main():
    ap = argparse.ArgumentParser(description="Connectivity RCA (CSV-driven)")
    ap.add_argument("--train", action="store_true", help="Train from CSV and save model")
    ap.add_argument("--data", type=str, default="data/processed/connectivity_dataset.csv", help="Path to CSV dataset")
    ap.add_argument("--model", type=str, default=MODEL_PATH, help="Path to save/load model")
    ap.add_argument("--predict", type=str, help="JSON string or @file.json with features")
    args = ap.parse_args()

    if args.train:
        train_and_save(args.data, args.model)
        return

    if args.predict:
        # allow @file.json
        if args.predict.startswith("@"):
            with open(args.predict[1:], "r") as f:
                feats = json.load(f)
        else:
            feats = json.loads(args.predict)
        res = predict_from_features(feats, args.model)
        print(json.dumps(res, indent=2))
        return

    print("Nothing to do. Use --train or --predict. Example:")
    print("  python connectivity_model.py --train --data data/processed/connectivity_dataset.csv")
    print('  python connectivity_model.py --predict \'{"rtt_gw_ms":2,"rtt_public_ms":280,"loss_public_pct":72,"dns_latency_ms":22,"tr_last_hop_idx":2,"rssi_dbm":-56,"dhcp_lease_age_min":900,"jitter_ms":18,"dns_ttl_avg":180,"dns_response_ip_diversity":1,"ping_gw_ok":"yes","ping_public_ok":"no","ping_dns_ip_ok":"no","dns_ok":"no","ip_valid":"yes","eth_link_up":"no","wifi_connected":"yes","error_code":"no_route"}\'')
    print("  python connectivity_model.py --predict @probe_output.json")

if __name__ == "__main__":
    main()
