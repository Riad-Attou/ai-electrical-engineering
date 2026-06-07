function pwm = servo_excitation(t, mode)
    
    % Servo excitation by pwm command in [-1, 1]. Two modes:
    % - mode = 0: chirp (sinesweep) used in training set
    % - mode = 1: multisine used in test set (unseen excitation)

    % --- Parameters ----------------------------------------------------
    p = servo_params();
    A = 0.9;

    % --- PWM design ----------------------------------------------------
    if mode < 0.5
        f0 = 1; f1 = 20; T = p.Tend;
        k  = (f1 - f0) / T;
        u  = A * sin(2*pi*(f0*t + 0.5*k*t^2));
    else
        freqs  = [1.3 2.7 4.1 6.5 9.2 12.8];
        phases = [0.10 1.74 3.05 5.21 2.66 4.88];
        u = 0;
        for n = 1:numel(freqs)
            u = u + sin(2*pi*freqs(n)*t + phases(n));
        end
        u = A * u / numel(freqs);
    end

    % Clip the result in [-1, 1]
    pwm = max(min(u, 1), -1);

end
