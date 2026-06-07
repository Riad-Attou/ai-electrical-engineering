function export_dataset(out, name)

    % Export the simulated dataset to as a csv file.
    %
    % The dataset has 10 columns, in this order. The tag shows how each 
    % signal is meant to be used for the learning task: [F] input feature, 
    % [F*] optional feature, [T] prediction target, [-] not for training:
    % - pwm      [F]   Normalized drive command in [-1, 1] 
    % - V        [-]   Applied motor voltage [V] 
    % - i        [-]   Armature current [A] 
    % - theta_m  [-]   Motor shaft angle [rad] 
    % - omega_m  [-]   Motor shaft angular velocity [rad/s] 
    % - theta_l  [T]   Load (output) shaft angle [rad] 
    % - omega_l  [-]   Load (output) shaft angular velocity [rad/s]   
    % - enc_m    [F]   Motor encoder reading [rad]                    
    % - enc_o    [F*]  Output encoder reading [rad]                  
    % - temp     [F*]  Motor temperature [C]                         
    %
    % Minimal feature set for the RNN-vs-Kalman experiment:
    % - inputs = [pwm, enc_m, (optionally + enc_o, temp)]
    % - target = theta_l
    % 
    % Attention: never feed ground-truth state (theta_m, omega_m, omega_l)
    % as inputs, that is exactly the information the NN must reconstruct.
    
    
    % --- Import data from Simulink -------------------------------------
    
    % Default file prefix if the input is empty
    if nargin < 2 || isempty(name)
        name = 'servo'; 
    end
 
    % Pull the logged signals from the simulation output.
    t = out.tout(:); % Time vector            [N x 1]
    M = out.simdata; % Logged signals (Array) [N x 10]

    % --- Build CSV file ------------------------------------------------
 
    % Define column names
    cols = {'t','pwm','V','i','theta_m','omega_m','theta_l','omega_l', ...
            'enc_m','enc_o','temp'};
 
    % Sanity check
    if size(M,2) ~= numel(cols)-1
        warning('Expected %d columns, found %d. Check the Mux.', ...
                 numel(cols)-1, size(M,2));
    end
 
    % Build a table and write the CSV
    T = array2table([t, M], 'VariableNames', cols);
    writetable(T, [name '.csv']);
 
    % Print a short report
    fprintf('Dataset "%s": %d samples, %d signals -> %s.csv\n', ...
            name, numel(t), size(M,2), name);
end
