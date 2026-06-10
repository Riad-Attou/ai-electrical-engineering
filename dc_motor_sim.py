import numpy as np
import pandas as pd
import os

# Very simple stable parameters
R   = 1.0
L   = 0.1
K_t = 0.01
K_b = 0.01
J   = 0.1
B   = 0.1
dt  = 0.0001   # ← smaller timestep = more stable
T   = 5.0
t   = np.arange(0, T, dt)

omega   = np.zeros(len(t))
current = np.zeros(len(t))
torque  = np.zeros(len(t))

V      = np.ones(len(t)) * 12.0
T_load = np.ones(len(t)) * 0.01

for k in range(1, len(t)):
    di = (V[k] - R*current[k-1] - K_b*omega[k-1]) / L
    dw = (K_t*current[k-1] - B*omega[k-1] - T_load[k]) / J
    current[k] = current[k-1] + di * dt
    omega[k]   = omega[k-1]   + dw * dt
    torque[k]  = K_t * current[k]

noise = 0.001
current_noisy = current + np.random.normal(0, noise, len(t))
omega_noisy   = omega   + np.random.normal(0, noise, len(t))

save_dir = '/Users/pasorn/Desktop/AI-Electrical/data'
os.makedirs(save_dir, exist_ok=True)
df = pd.DataFrame({
    'time':          t,
    'voltage':       V,
    'current_true':  current,
    'current_noisy': current_noisy,
    'omega_true':    omega,
    'omega_noisy':   omega_noisy,
    'torque_true':   torque,
    'T_load':        T_load
})
df.to_csv(os.path.join(save_dir, 'motor_data.csv'), index=False)

print(f"✅ Data generated! {len(t)} samples", flush=True)
print(f"   Max speed  : {omega.max():.4f} rad/s",   flush=True)
print(f"   Max current: {current.max():.4f} A",     flush=True)
print(f"   Max torque : {torque.max():.4f} Nm",     flush=True)