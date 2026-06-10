import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

save_dir = '/Users/pasorn/Desktop/AI-Electrical/data'
os.makedirs(save_dir, exist_ok=True)

R   = 1.0
L   = 0.1
K_t = 0.01
K_b = 0.01
J   = 0.1
B   = 0.1
dt  = 0.0001   # ← must match simulation!

df          = pd.read_csv(os.path.join(save_dir, 'motor_data.csv'))
V           = df['voltage'].values
z           = df['current_noisy'].values
omega_true  = df['omega_true'].values
torque_true = df['torque_true'].values

# Build F and check it's stable (all eigenvalues < 1)
F = np.array([
    [1 - B/J  * dt,   K_t/J * dt],
    [-K_b/L   * dt,   1 - R/L * dt]
])
eigenvalues = np.abs(np.linalg.eigvals(F))
print(f"F eigenvalues: {eigenvalues}", flush=True)
print(f"Stable: {all(eigenvalues < 1)}", flush=True)

B_mat   = np.array([[0.0], [dt/L]])
H       = np.array([[0, 1]])
x       = np.array([[0.0], [0.0]])
P       = np.eye(2) * 0.1
Q       = np.eye(2) * 1e-6
R_noise = np.array([[0.001]])

omega_kf  = np.zeros(len(V))
torque_kf = np.zeros(len(V))

for k in range(len(V)):
    # Predict
    x = F @ x + B_mat * V[k]
    P = F @ P @ F.T + Q

    # Update
    y = z[k] - (H @ x)[0, 0]
    S = H @ P @ H.T + R_noise
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K * y
    P = (np.eye(2) - K @ H) @ P

    omega_kf[k]  = x[0, 0]
    torque_kf[k] = K_t * x[1, 0]

df['omega_kf']  = omega_kf
df['torque_kf'] = torque_kf
df.to_csv(os.path.join(save_dir, 'motor_data.csv'), index=False)

rmse_omega  = np.sqrt(np.mean((omega_true  - omega_kf)  ** 2))
rmse_torque = np.sqrt(np.mean((torque_true - torque_kf) ** 2))
print(f"✅ Kalman Filter Done!", flush=True)
print(f"   RMSE Speed  : {rmse_omega:.6f} rad/s", flush=True)
print(f"   RMSE Torque : {rmse_torque:.6f} Nm",   flush=True)

# Plot only first 1000 steps for clarity
n = 10000
time = df['time'].values

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(time[:n], omega_true[:n], label='True Speed',  linewidth=2)
axes[0].plot(time[:n], omega_kf[:n],   label='KF Estimate', linestyle='--')
axes[0].set_title('Motor Speed Estimation - Kalman Filter')
axes[0].set_ylabel('Speed (rad/s)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(time[:n], torque_true[:n], label='True Torque', linewidth=2)
axes[1].plot(time[:n], torque_kf[:n],   label='KF Estimate', linestyle='--')
axes[1].set_title('Motor Torque Estimation - Kalman Filter')
axes[1].set_ylabel('Torque (Nm)')
axes[1].set_xlabel('Time (s)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kalman_results.png'))
plt.show()
print("📊 Plot saved!", flush=True)