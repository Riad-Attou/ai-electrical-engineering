# ⚡ Motor Speed & Torque Estimation
### AI for Electrical Engineering — Group Project

A hybrid dynamic estimation system for DC motor speed and torque using **Kalman Filter + LSTM Neural Network** in Python.

---

## 👥 Team Members
| Name | Role |
|------|------|
| Member 1 | Simulation & Data Generation |
| Member 2 | Kalman Filter Implementation |
| Member 3 | LSTM Neural Network |
| Member 4 | Hybrid Estimator & Evaluation |

> ✏️ Replace with your actual names and roles!

---

## 📌 Project Overview

In real DC motors, speed and torque sensors are expensive and noisy.
This project estimates them **without direct sensors** using only:
- ✅ Voltage (V)
- ✅ Current (i)

We compare three approaches:

| Method | Description |
|--------|-------------|
| Kalman Filter | Classical model-based estimator |
| LSTM | Deep learning time-series estimator |
| **Hybrid (KF + LSTM)** | Best of both — KF + LSTM error correction |

---

## 🗂️ Project Structure

```
AI-Electrical/
│
├── dc_motor_sim.py        # Step 1: Simulate motor & generate data
├── kalman_filter.py       # Step 2: Classical KF estimation
├── lstm_model.py          # Step 3: LSTM neural network
├── hybrid_estimator.py    # Step 4: Hybrid KF + LSTM
├── metrics.py             # Step 5: Evaluation & comparison plots
│
├── data/                  # Auto-generated data folder
│   ├── motor_data.csv     # Simulated motor dataset
│   ├── lstm_model.pth     # Saved LSTM model
│   ├── scaler_X.pkl       # Feature scaler
│   ├── scaler_y.pkl       # Target scaler
│   ├── kalman_results.png
│   ├── lstm_results.png
│   └── hybrid_results.png
│
└── README.md
```

---

## ⚙️ DC Motor Model

**Electrical equation:**
```
V = R·i + L·(di/dt) + Kb·ω
```

**Mechanical equation:**
```
J·(dω/dt) = Kt·i - B·ω - T_load
```

| Symbol | Meaning | Value |
|--------|---------|-------|
| R | Resistance | 1.0 Ω |
| L | Inductance | 0.1 H |
| Kt | Torque constant | 0.01 Nm/A |
| Kb | Back-EMF constant | 0.01 V·s/rad |
| J | Moment of inertia | 0.1 kg·m² |
| B | Friction coefficient | 0.1 Nm·s/rad |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/AI-Electrical.git
cd AI-Electrical
```

### 2. Install Dependencies
```bash
pip install numpy pandas matplotlib scikit-learn torch joblib
```

### 3. Run the Project (in order!)
```bash
# Step 1 — Generate simulation data
python dc_motor_sim.py

# Step 2 — Run Kalman Filter
python kalman_filter.py

# Step 3 — Train LSTM
python lstm_model.py

# Step 4 — Run Hybrid Estimator
python hybrid_estimator.py

# Step 5 — Final evaluation & comparison
python metrics.py
```

---

## 🧠 How It Works

```
Input Signals: [Voltage V, Current i]
        │
        ├──────────────────────────────┐
        ▼                              ▼
  Kalman Filter                  LSTM Network
  (model-based)                  (data-driven)
        │                              │
        └──────────┬───────────────────┘
                   ▼
           Hybrid Estimator
        (KF estimate + LSTM correction)
                   │
                   ▼
        Final: [ω_estimated, T_estimated]
```

### Kalman Filter
- Uses motor physics equations
- Predict → Update loop every timestep
- Fast and interpretable

### LSTM Neural Network
- Looks back 20 timesteps (sequence learning)
- Input: `[V, i, ω_kf, T_kf]`
- Output: corrected `[ω, T]`
- 2-layer LSTM with dropout

### Hybrid System
- KF provides initial estimate
- LSTM corrects the KF error
- Combined output is more accurate than either alone

---

## 📊 Results

| Method | RMSE Speed (rad/s) | RMSE Torque (Nm) |
|--------|--------------------|------------------|
| Kalman Filter | — | — |
| LSTM Only | — | — |
| **Hybrid (KF+LSTM)** | **—** | **—** |

> 📝 Fill in your actual results after running all scripts!

---

## 📦 Dependencies

```
numpy
pandas
matplotlib
scikit-learn
torch
joblib
```

Install all at once:
```bash
pip install numpy pandas matplotlib scikit-learn torch joblib
```

---

## 📁 Git Tips for the Team

### First time setup
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AI-Electrical.git
git push -u origin main
```

### Daily workflow
```bash
# Before you start working
git pull origin main

# After you make changes
git add .
git commit -m "describe what you changed"
git push origin main
```

### Each member works on their own branch
```bash
# Create your branch
git checkout -b your-name/feature-name

# Example
git checkout -b ahmed/lstm-model

# Push your branch
git push origin ahmed/lstm-model
```

---

## 🙏 Acknowledgements
- Course: AI for Electrical Engineering
- Method: Hybrid Kalman Filter + LSTM
- Framework: PyTorch

---

## 📄 License
This project is for academic purposes only.
