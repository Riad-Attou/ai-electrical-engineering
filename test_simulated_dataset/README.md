# Servo with Gearbox Data Generator: Motor Paramters Identification and Backlash Compensation

A didactic, simulation-only toolchain in MATLAB/Simulink.
It simulates a realistic servo (DC motor plus a geared output with backlash) and produces `.csv` datasets for two machine-learning experiments:

1. **Motor Parameter Identification** (ML1): replaces the linear motor identification with a learned nonlinear model (SINDy).
2. **Backlash compensation** (ML2): estimates the true output angle through the play with a recurrent network (GRU), compared against a linear Kalman-class baseline.

The plant is the same for both. Only the configuration and the columns you use change.

---

## 1. Plant physics

A two-inertia model (motor and load) coupled by a reduction gear with compliant backlash. The state integrated in `servo_plant_ode.m` is

```
x = [ i ; omega_m ; theta_m ; omega_l ; theta_l ]
```

armature current, motor velocity and angle, load (output) velocity and angle.

### Drive (pwmNonlin)

The `pwm` command in [-1, 1] passes through a deadband and becomes a voltage:

$$ pn = \mathrm{sign}(pwm)\,\frac{\max(|pwm| - d_b,\, 0)}{1 - d_b}, \qquad V = pn \cdot V_{max} $$

with deadband $d_b = 0.08$. This represents the dead zone of an H-bridge at low duty cycle.

### Electrical

$$ L\,\frac{di}{dt} = V - R\,i - K_e\,\omega_m, \qquad T_{em} = K_t\,i $$

The electrical time constant $L/R = 2.5\cdot10^{-4}$ s is far faster than the mechanical one, so the current is quasi-static: $i \approx (V - K_e\omega_m)/R$.

### Friction (Stribeck) and cogging

$$ T_{f,m} = \Big[\big(F_{c,m} + (F_{s,m}-F_{c,m})e^{-(\omega_m/v_s)^2}\big)\tanh\!\frac{\omega_m}{10^{-3}} + b_m\,\omega_m\Big]\,k_f $$

$$ T_{cog} = T_{cog}\,\sin(N_{cog}\,\theta_m) $$

`tanh(omega/1e-3)` is a smooth approximation of `sign`, which avoids the discontinuity at zero velocity.

### Mesh with backlash (two inertias)

Transmission error referred to the motor, and the backlash dead zone:

$$ \varphi = \theta_m - N\,\theta_l, \qquad dz = \varphi - \mathrm{clamp}(\varphi,\, -g,\, +g) $$

$$ T_{mesh} = (1 - decouple)\big[\,k_g\,dz + c_g\,(\omega_m - N\,\omega_l)\,\big] $$

with $g = $ `gap_motor` (half-backlash referred to the motor). Inside the band $|\varphi| \le g$, $dz = 0$: the motor turns against no load (free-play), and outside the band the gear engages. This dead zone produces the **hysteresis** that the linear baseline cannot represent.

### Motor and load mechanics

$$ J_m\,\frac{d\omega_m}{dt} = T_{em} - T_{f,m} - T_{cog} - T_{mesh}, \qquad \frac{d\theta_m}{dt} = \omega_m $$

$$ J_l\,\frac{d\omega_l}{dt} = N\,\eta\,T_{mesh} - T_{f,l} - T_{ext}, \qquad \frac{d\theta_l}{dt} = \omega_l $$

with $T_{f,l} = (F_{c,l}\tanh(\omega_l/10^{-3}) + b_l\,\omega_l)\,k_f$.

### Thermal drift (warm-up)

$$ T(t) = T_{amb} + \Delta T_{warm}\,(1 - e^{-t/\tau_{th}}), \qquad k_f = 1 + k_{T,fric}\,(T - T_{amb}) $$

Friction falls as the motor warms ($k_{T,fric} < 0$); the backlash grows ($k_{T,gap} > 0$).

### Sensors (dual encoder)

$$ enc_m = \mathrm{round}\!\Big(\frac{\theta_m + n_m}{q_m}\Big)q_m, \qquad q_m = \frac{2\pi}{2^{enc\_m\_bits}} $$

$$ \theta_{l,nl} = \theta_l + h_1\sin\theta_l + h_2\sin 2\theta_l, \qquad enc_o = \mathrm{round}\!\Big(\frac{\theta_{l,nl} + n_o}{q_o}\Big)q_o $$

The output encoder (AS5048a) carries a periodic nonlinearity (eccentricity) on top of noise and quantisation.

---

## 2. The equations behind the two ML tasks

### Motor parameter identification

You identify the motor velocity dynamics:

$$ \frac{d\omega_m}{dt} = a_{cont}\,\omega_m + b_{cont}\,pwmNonlin(u) + c_{fric}\,\mathrm{sign}(\omega_m) + c_{EMF}\,|pwmNonlin(u)|\,\omega_m $$

In the decoupled run ($T_{mesh} = 0$), with quasi-static current, the motor obeys

$$ \frac{d\omega_m}{dt} \approx \frac{K_t}{R\,J_m}V - \frac{K_t K_e}{R\,J_m}\omega_m - \frac{T_{f,m} + T_{cog}}{J_m} $$

The technique to use here is **SINDy** (sparse identification of nonlinear dynamics): from `(omega_m, V)` you fit `d(omega_m)/dt` against a small library of candidate terms (polynomials, a `sign`-like term for Coulomb friction, a `|V|*omega_m` term for back-EMF) and keep only the few that survive a sparse regression. You get an interpretable model whose coefficients are $a_{cont}$, $b_{cont}$, $c_{fric}$, $c_{EMF}$, the values the pole-placement gains depend on. No code implements this yet; the generator produces the dataset that such a routine would consume.

### Backlash compensation

The hysteresis comes from the dead zone $dz$: the map $\theta_l$ versus $\theta_m/N$ has a lost-motion loop whose width equals the backlash. A linear observer (Kalman class) assumes rigid coupling $\theta_l \approx \theta_m/N$ and makes an error at every reversal The technique to use here is a **recurrent network** (GRU or LSTM): its hidden state carries the memory the lost-motion loop needs, so from the history of `[pwm, enc_m]` it can estimate the true $\theta_l$ where a linear observer cannot. Compare it against a linear Kalman-class baseline on the same inputs. No code implements this yet; the generator produces the train and test datasets such a network would consume.

---

## 3. Setting `servo_params` for the two ML tasks

`p.decouple` selects the plant configuration: keep it at `0` for the coupled plant, set it to `1` to free the motor for identification. `servo_plant_ode` re-reads `servo_params` at every step, so a changed value takes effect on the next run, with no rebuild.

| Parameter | Motor parameter identification | Backlash compensation |
|-----------|-------------------------------|-----------------------|
| `p.decouple` | **1** (motor free, clean dynamics) | **0** (full coupled plant) |
| `p.Tcog` | `0` recommended (cleaner identification) | nominal value `2e-3` |
| `Mode` block | **0** to fit, **1** to validate | **0** for train, **1** for test |
| features used | `pwm` (or `V`), `omega_m` | `pwm`, `enc_m` (opt. `enc_o`, `temp`) |
| target | `d(omega_m)/dt` (from `omega_m`) | `theta_l` |
| export | `export_dataset(out,'ident_train')` / `'ident_test'` | `export_dataset(out,'train')` / `'test'` |

For backlash compensation, do **not** feed the true states `theta_m`, `omega_m`, `omega_l` as inputs: that is the information the network has to reconstruct (leakage). For motor identification, those motor-side signals are the data.

---

## 4. Difference between the training set and the test set

This applies to **both tasks**. The physics stays **identical** (same `servo_params`, with `p.decouple`fixed for the task). Only the excitation changes: the `Mode` block selects it, and `servo_excitation.m` implements it.

| | Mode | Signal | Purpose |
|---|------|---------|-------|
| **Training** | `0` | chirp 1 → 20 Hz, amplitude 0.9 | covers the spectrum with no gaps |
| **Test** | `1` | multisine (6 tones: 1.3 … 12.8 Hz) | excitation **never seen** in training |

You fit the model on the chirp and check it on the multisine. The input differs but the system is the same, so the test error measures **generalisation** rather than memorisation of one trajectory. For motor identification you simulate the identified equation forward on the multisine and compare to ground truth; for backlash compensation you compare the estimate of $\theta_l$ to ground truth. For a harder test (distribution shift) you can also change the seed of the Random blocks, or add a disturbance `p.Text`, leaving everything else fixed. The chirp/multisine split is the baseline.

---

## 5. Files and run order

| File | Role |
|------|-------|
| `servo_params.m` | physical parameters (struct `p`) |
| `servo_plant_ode.m` | full physics (state derivatives) |
| `servo_excitation.m` | pwm command: chirp (train) / multisine (test) |
| `servo_sensors.m` | the two encoder measurements |
| `build_servo_compact.m` | builds the `dataset_simulation` model in code |
| `export_dataset.m` | exports `out.simdata` to `.csv` |

```matlab
build_servo_compact                          % once

% --- Motor parameter identification ---  (p.decouple = 1; p.Tcog = 0;)
% Mode block = 0:  (fit)
out = sim('dataset_simulation');  export_dataset(out,'ident_train');
% Mode block = 1:  (validate)
out = sim('dataset_simulation');  export_dataset(out,'ident_test');

% --- Backlash compensation ---  (set p.decouple = 0; again)
% Mode block = 0:
out = sim('dataset_simulation');  export_dataset(out,'train');
% Mode block = 1:
out = sim('dataset_simulation');  export_dataset(out,'test');
```

The machine-learning step is **not implemented yet**. The generator only produces the CSV datasets: a SINDy routine would read `ident_train`/`ident_test`, and a recurrent-network observer would read `train`/`test`.

### CSV schema (10 columns plus time)

`t, pwm, V, i, theta_m, omega_m, theta_l, omega_l, enc_m, enc_o, temp`

Usage tags for backlash compensation: `pwm`, `enc_m` are features; `enc_o`, `temp` are optional; `theta_l` is the target; `V`, `i`, `theta_m`, `omega_m`, `omega_l` are ground truth (not for use as inputs). For motor identification the motor-side signals (`pwm`/`V`, `omega_m`) are the data instead.

---

## Notes

- Recommended solver: `ode23t`, `MaxStep = dt/2`.
- The build scripts are **untested** (MATLAB does not run here): if a block name differs in your
  release, fix it in the flagged block; `Ctrl+Shift+A` rearranges the layout.
- Codegen rule: define every struct field before you read any of them; compute derived parameters
  from local variables (see `gap_motor`).
- Parameter values live in `servo_params.m`, didactic but physically consistent.
