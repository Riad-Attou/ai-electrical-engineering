function [enc_m, enc_o] = servo_sensors(thm, thl, nm, no)
    
    % Dual-encoder (cascade loop) measurements from the true angles.
    % Add noises nm and no provided by Random Number blocks:
    % - Motor encoder  : noise + quantization
    % - Output encoder : periodic nonlinearity + noise + quantization

    % --- Apply quantization --------------------------------------------
    p  = servo_params();
    qm = 2*pi / 2^p.enc_m_bits;
    qo = 2*pi / 2^p.enc_o_bits;

    % Apply non-ideality ------------------------------------------------
    
    % Motor encoder
    enc_m = floor((thm + nm)/qm + 0.5) * qm;

    % Output encoder
    thl_nl = thl + p.enc_o_h1*sin(thl) + p.enc_o_h2*sin(2*thl);
    enc_o  = floor((thl_nl + no)/qo + 0.5) * qo;

end

