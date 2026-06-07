function [dx, aux] = servo_plant_ode(pwm, t, x)
    
    % State derivatives of the realistic servo (full physics):
    % - x   = [ i ; omega_m ; theta_m ; omega_l ; theta_l]
    % - dx  = state derivatives
    % - aux = [ V ; temp ]   (for scopes/logging)
    %
    % Parameters are read from servo_params at every step

    p = servo_params();

    i  = x(1);  
    wm = x(2);  
    thm = x(3);  
    wl = x(4);  
    thl = x(5);

    % --- Drive ---------------------------------------------------------
    pwm_s = max(min(pwm, 1), -1);
    db    = 0.08;
    pn    = sign(pwm_s) * max(abs(pwm_s) - db, 0) / (1 - db);
    pn    = max(min(pn, 1), -1);
    V     = pn * p.Vmax;

    % --- Thermal (warm-up modulates friction) --------------------------
    temp = p.T_amb + p.dT_warm * (1 - exp(-t / p.tau_th));
    kf   = 1 + p.kT_fric * (temp - p.T_amb);

    % --- Electrical ----------------------------------------------------
    di   = (V - p.R*i - p.Ke*wm) / p.L;
    T_em = p.Kt * i;

    % --- Motor friction (Stribeck * thermal + cogging) -----------------
    Tf_m  = ((p.Fc_m + (p.Fs_m - p.Fc_m) * exp(-(wm/p.vs)^2)) * ...
        tanh(wm/1e-3) + p.bm*wm) * kf;
    T_cog = p.Tcog * sin(p.Ncog * thm);

    % --- Two-inertia mesh with compliant backlash ----------------------
    phi    = thm - p.N*thl;
    dz     = phi - max(min(phi, p.gap_motor), -p.gap_motor); % dead zone 
    T_mesh = (1 - p.decouple) * (p.kg*dz + p.cg*(wm - p.N*wl));

    % --- Motor mechanics -----------------------------------------------
    dwm  = (T_em - Tf_m - T_cog - T_mesh) / p.Jm;
    dthm = wm;

    % --- Load ----------------------------------------------------------
    Tf_l = (p.Fc_l * tanh(wl/1e-3) + p.bl * wl) * kf;
    dwl  = (p.N*p.eta*T_mesh - Tf_l - p.Text) / p.Jl;
    dthl = wl;

    dx  = [di; dwm; dthm; dwl; dthl];
    aux = [V; temp];
    
end
