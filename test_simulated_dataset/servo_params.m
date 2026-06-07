function p = servo_params()
    
    % Physical parameters used in the data generator for servo and gearbox
    % 
    % Two-inertia model with compliant backlash, Stribeck friction, 
    % electrical dynamics, cogging, non-ideal sensors and thermal drift.
    %
    % Two physical coordinates, motor on the motor side, load on the 
    % output side; the gear ratio N appears explicitly at the mesh.

    % --- System setup --------------------------------------------------
    
    % 0 = coupled plant (Used for backlash compensation) 
    % 1 = free motor (Used for motor paramter identification)
    p.decouple = 0;  

    % --- Sampling ------------------------------------------------------
    p.dt    = 1e-3;         % Sensor/control time step [s]
    p.Tend  = 30.0;         % Run duration [s]

    % --- Electrical (Armature) -----------------------------------------
    p.R     = 2.0;          % Resistance [ohm]
    p.L     = 5e-4;         % Inductance [H]
    p.Ke    = 0.05;         % Back-EMF constant [V/(rad/s)]
    p.Kt    = 0.05;         % Torque constant [Nm/A]
    p.Vmax  = 12.0;         % Ssupply voltage [V]

    % --- Motor (Motor side) --------------------------------------------
    p.Jm    = 2e-5;         % Rotor inertia [kg m^2]
    p.bm    = 1e-4;         % Motor viscous friction [Nm/(rad/s)]
    p.Fc_m  = 3e-3;         % Motor Coulomb friction [Nm]
    p.Fs_m  = 6e-3;         % Motor static/break-away friction [Nm]
    p.vs    = 5.0;          % Stribeck velocity [rad/s]

    % --- Cogging torque ------------------------------------------------
    p.Tcog  = 2e-3;         % Cogging amplitude [Nm]
    p.Ncog  = 12;           % Cogging periods per revolution [-]

    % --- Gearbox -------------------------------------------------------
    N        = 50; 
    p.N      = N;           % Reduction ratio [-]
    p.eta    = 0.85;        % Efficiency [-]
    gap_out  = 0.02;
    p.gap_out= gap_out;     % Output-side half-backlash [rad]
    p.kg     = 5.0;         % Mesh torsional stiffness motor side [Nm/rad]
    p.cg     = 0.02;        % Mesh damping [Nm/(rad/s)]
    
    % --- Load (Output side) --------------------------------------------
    p.Jl     = 5e-3;        % Load inertia [kg m^2]
    p.bl     = 5e-3;        % Load viscous friction [Nm/(rad/s)]
    p.Fc_l   = 1e-2;        % load Coulomb friction [Nm]
    p.Text   = 0.0;         % External/disturbance torque [Nm]

    % --- Sensors (Two encoder) -----------------------------------------
    p.enc_m_bits  = 12;     % Motor encoder (optical) [bits/rev]
    p.enc_o_bits  = 14;     % Output encoder AS5048a [bits/rev]
    p.enc_m_noise = 2e-4;   % Motor encoder noise [rad rms]
    p.enc_o_noise = 1e-4;   % Output encoder noise [rad rms]
    p.enc_o_h1    = 8e-4;   % First harmonic error (eccentricity) [rad]
    p.enc_o_h2    = 3e-4;   % Second harmonic error [rad]
    p.enc_lat     = 0.5e-3; % Output read latency [s]

    % --- Thermal drift (Warm-up) ---------------------------------------
    p.T_amb    = 25;        % Ambient temperature [C]
    p.dT_warm  = 30;        % Rise to steady state [C]
    p.tau_th   = 3.0;       % Thermal time constant [s]
    p.kT_fric  = -0.01;     % Friction coefficient decreases with T [1/C]
    p.kT_gap   = +0.005;    % Backlash coefficient grows with T [1/C]

    % --- Backlash ------------------------------------------------------
    
    % Derived half-backlash reflected on motor side [rad]
    p.gap_motor = N * gap_out; 
    
end
