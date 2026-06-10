# evaluation/metrics.py
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def evaluate(y_true, y_pred, label="Speed"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = np.mean(np.abs(y_true - y_pred))
    
    print(f"--- {label} Estimation ---")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="True", linewidth=2)
    plt.plot(y_pred, label="Estimated", linestyle="--")
    plt.title(f"{label} Estimation")
    plt.xlabel("Time Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()