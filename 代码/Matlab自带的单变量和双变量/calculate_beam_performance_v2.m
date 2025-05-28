% --- Core Engineering Model: Calculate Mass and F_ext (Version 3 - Ternary Op Fixed) ---
function [mass, F_ext_final, varargout] = calculate_beam_performance_v2(x_reduced, params)
    % Unpack design variables [L1, R1, L2, R2]
    L1 = x_reduced(1); R1 = x_reduced(2); L2 = x_reduced(3); R2 = x_reduced(4);

    % Unpack FIXED dimensions and other parameters
    L_total = params.L_total_fixed; % FIXED
    B = params.B_fixed;             % FIXED
    rho = params.rho; E = params.E; sigma_limit = params.sigma_limit;
    D_limit = params.D_limit; W_fixed = params.W_fixed; g = params.g;

    % --- 1. Calculate Mass ---
    A_base = (L_total * B) - pi * (R1^2 + R2^2);
    mass = rho * W_fixed * A_base;

    % --- 2. Calculate Max Allowable F_ext ---
    I_gross = (W_fixed * B^3) / 12;
    w_self = rho * g * W_fixed * B;

    delta_self_weight = inf; 
    if E > 0 && I_gross > 1e-12 
        delta_self_weight = (w_self * L_total^4) / (8 * E * I_gross);
    end
    
    F_max_deform = 0;
    if L_total > 1e-6 && E > 0 && I_gross > 1e-12 && D_limit > delta_self_weight
        F_max_deform = ((D_limit - delta_self_weight) * 3 * E * I_gross) / (L_total^3);
    end
    F_max_deform = max(0, F_max_deform);

    % Stress Checks
    S_gross = 0; F_max_stress_fixed = 0;
    if B > 1e-6 && L_total > 1e-6
        S_gross = (W_fixed * B^2) / 6;
        M_sw_fixed = (w_self * L_total^2) / 2;
        if S_gross > 1e-12 
            sigma_sw_fixed_end = M_sw_fixed / S_gross;
            if sigma_limit > sigma_sw_fixed_end
                F_max_stress_fixed = ((sigma_limit - sigma_sw_fixed_end) * S_gross) / L_total;
            end
        else
            F_max_stress_fixed = 0; 
        end
    end
    F_max_stress_fixed = max(0, F_max_stress_fixed);

    Kt_func = @(Rad, B_beam) min(max(3.0 - 3.14*(2*Rad/B_beam) + 3.67*(2*Rad/B_beam)^2 - 1.53*(2*Rad/B_beam)^3, 1.0), 5.0);

    % Hole 1
    F_max_stress_h1 = 0; % Default to 0
    L_eff1 = L_total - L1; B_net1 = B - 2*R1;
    if B_net1 > 1e-6 && W_fixed > 1e-6 
        S_net1 = (W_fixed * B_net1^2) / 6; Kt1 = Kt_func(R1, B);
        M_sw_h1 = (w_self * L_eff1^2) / 2;
        
        sigma_sw_nom_h1 = inf; % Default if S_net1 is too small
        if S_net1 > 1e-12
            sigma_sw_nom_h1 = M_sw_h1 / S_net1;
        end

        if L_eff1 <= 1e-6 % Hole at or beyond free end
            if Kt1 > 1e-6 && (sigma_limit / Kt1) <= sigma_sw_nom_h1 
                F_max_stress_h1 = 0; % Self-weight alone causes failure
            else
                F_max_stress_h1 = inf; % F_ext causes no (tensile) bending moment here
            end
        elseif Kt1 > 1e-6 && (sigma_limit / Kt1) > sigma_sw_nom_h1 
            F_max_stress_h1 = ((sigma_limit / Kt1 - sigma_sw_nom_h1) * S_net1) / L_eff1;
        else 
            % This case implies Kt1 is too small (should be >=1 from Kt_func) 
            % OR self-weight stress alone (or with minimal F_ext) exceeds limit.
            F_max_stress_h1 = 0; % Cannot bear any F_ext
        end
    end
    F_max_stress_h1 = max(0, F_max_stress_h1); % Ensure non-negative
    
    % Hole 2
    F_max_stress_h2 = 0; % Default to 0
    L_eff2 = L_total - L2; B_net2 = B - 2*R2;
    if B_net2 > 1e-6 && W_fixed > 1e-6 
        S_net2 = (W_fixed * B_net2^2) / 6; Kt2 = Kt_func(R2, B);
        M_sw_h2 = (w_self * L_eff2^2) / 2;

        sigma_sw_nom_h2 = inf; % Default if S_net2 is too small
        if S_net2 > 1e-12
            sigma_sw_nom_h2 = M_sw_h2 / S_net2;
        end
        
        if L_eff2 <= 1e-6 % Hole at or beyond free end
            if Kt2 > 1e-6 && (sigma_limit / Kt2) <= sigma_sw_nom_h2 
                F_max_stress_h2 = 0; % Self-weight alone causes failure
            else
                F_max_stress_h2 = inf; % F_ext causes no (tensile) bending moment here
            end
        elseif Kt2 > 1e-6 && (sigma_limit / Kt2) > sigma_sw_nom_h2 
            F_max_stress_h2 = ((sigma_limit / Kt2 - sigma_sw_nom_h2) * S_net2) / L_eff2;
        else 
            F_max_stress_h2 = 0; % Cannot bear any F_ext
        end
    end
    F_max_stress_h2 = max(0, F_max_stress_h2); % Ensure non-negative

    F_max_stress = min([F_max_stress_fixed, F_max_stress_h1, F_max_stress_h2]);
    F_ext_final = min(F_max_deform, F_max_stress);
    F_ext_final = max(0, F_ext_final); % Must be non-negative

    if nargout > 2
        varargout{1} = F_max_deform; varargout{2} = F_max_stress;
    end
end