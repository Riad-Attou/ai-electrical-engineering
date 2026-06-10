# simulation/dc_motor_sim.py
import numpy as np
import pandas as pd
import os

save_dir = '/Users/pasorn/Desktop/AI-Electrical/data'
os.makedirs(save_dir, exist_ok=True)

# Motor parameters
R = 1.0      # Resistance (Ohm)
L = 0.5      # Inductance (H)
K_t = 0.01   # Torque constant
K_b = 0.01   # Back-EMF constant
J = 0.01     # Moment of inertia
B = 0.1      # Friction coefficient

dt = 0.001   # Time step
T = 5.0      # Total time (seconds)
t = np.arange(0, T, dt)

# State variables
omega = np.zeros(len(t))   # Speed
current = np.zeros(len(t)) # Current
torque = np.zeros(len(t))  # Torque

# Input voltage (step + noise)
V = 12 * np.ones(len(t))
T_load = 0.5 * np.ones(len(t))  # Constant load

noise_std = 0.05

for k in range(1, len(t)):
    di = (V[k] - R*current[k-1] - K_b*omega[k-1]) / L
    dw = (K_t*current[k-1] - B*omega[k-1] - T_load[k]) / J

    current[k] = current[k-1] + di * dt
    omega[k]   = omega[k-1]   + dw * dt
    torque[k]  = K_t * current[k]

# Add sensor noise
omega_noisy   = omega   + np.random.normal(0, noise_std, len(t))
current_noisy = current + np.random.normal(0, noise_std, len(t))

# Save
df = pd.DataFrame({
    'time': t,
    'voltage': V,
    'current_noisy': current_noisy,
    'omega_true': omega,
    'omega_noisy': omega_noisy,
    'torque_true': torque
})
df.to_csv(os.path.join(save_dir, 'motor_data.csv'), index=False)
print("Data generated!")