# Network_Troubleshooter
# AI/ML-Based Connectivity Troubleshooter  
### Minimal Proof-of-Concept Model for Network Fault Classification  

This repository contains a minimal machine learning prototype that diagnoses the **root cause of network connectivity failures** based on diagnostic telemetry (ping, traceroute, DNS, link status, etc.).  
It aligns with the AI/ML system proposed in the *Connectivity Troubleshooter (Use Case 32)* midterm report.

---

## 📁 Project Structure
├── connectivity_model.py # Main training and inference script
├── data/
│ └── processed/
│ └── connectivity_dataset.csv # Example dataset (sample diagnostic data)
├── connectivity_model.joblib # Trained model (generated after training)
└── README.md

yaml
Copy code

---

## ⚙️ Requirements
- Python 3.8+
- Required libraries:
  ```bash
  pip install scikit-learn pandas numpy joblib
