function run_beam_optimization_v4 % Renamed main function
% =========================================================================
% MAIN SCRIPT for Multi-Objective Cantilever Beam Optimization (Version 4)
% =========================================================================
% - Uses custom_lbfgs_with_penalty for single-objective optimization.

clear; close all; clc;
disp('Starting Cantilever Beam Optimization (Version 4 - Custom L-BFGS)...');

% --- Define Fixed Parameters and Material Properties ---
params.rho = 7850;          % Material Density (kg/m^3)
params.E = 200e9;           % Young's Modulus (Pa)
params.sigma_limit = 250e6; % Allowable Material Stress (Pa)
params.D_limit = 0.0035;    % Maximum Allowable Deflection (m)
params.W_fixed = 0.02;      % Fixed Beam Thickness (into the page) (m)
params.g = 9.81;            % Gravitational Acceleration (m/s^2)

params.L_total_fixed = 1.2; 
params.B_fixed = 0.12;      
disp(['FIXED Beam Dimensions: L_total = ', num2str(params.L_total_fixed), 'm, B = ', num2str(params.B_fixed), 'm']);

params.F_ext_min_for_mass_opt = 1000; 
params.M_max_for_Fext_opt = 25.0;     
disp(['Constraint for Max F_ext opt: Mass <= ', num2str(params.M_max_for_Fext_opt), ' kg']);
disp(['Constraint for Min Mass opt: F_ext >= ', num2str(params.F_ext_min_for_mass_opt), ' N']);

params.min_end_dist = 0.01;   
params.min_hole_sep = 0.01;   
params.min_ligament = 0.005;  
params.tol = 1e-6;            

params.nvars = 4; 

% --- Design Variable Bounds [L1, R1, L2, R2] ---
R_min_val = 0.005; 
R_max_val = params.B_fixed/2 - params.min_ligament - params.tol;
if R_max_val <= R_min_val, error('R_max_val too small.'); end

L1_min_val = params.min_end_dist + R_max_val; 
L1_max_val = params.L_total_fixed * 0.45; 
if L1_max_val <= L1_min_val, L1_max_val = L1_min_val + 0.1 * params.L_total_fixed; warning('L1_max_val adjusted.'); end
if L1_max_val >= params.L_total_fixed - params.min_end_dist - R_min_val, L1_max_val = params.L_total_fixed - params.min_end_dist - R_min_val - 1e-3; end

L2_min_val = params.L_total_fixed * 0.55; 
L2_max_val = params.L_total_fixed - params.min_end_dist - R_max_val; 
if L2_max_val <= L2_min_val, L2_max_val = L2_min_val + 0.1 * params.L_total_fixed; warning('L2_max_val adjusted.'); end
L2_min_candidate = L1_max_val + R_max_val + params.min_hole_sep + R_min_val;
if L2_min_val < L2_min_candidate
    L2_min_val = L2_min_candidate;
    if L2_min_val >= L2_max_val, L2_max_val = L2_min_val + 0.1 * params.L_total_fixed; warning('L2 bounds re-adjusted.'); end
end

lb = [L1_min_val, R_min_val, L2_min_val, R_min_val];
ub = [L1_max_val, R_max_val, L2_max_val, R_max_val];

for i=1:params.nvars
    if lb(i) >= ub(i), error('Bound error: lb(%d) >= ub(%d).', i, i); end
end

% --- Initial Guess (for custom L-BFGS) [L1, R1, L2, R2] ---
x0 = [ (lb(1) + ub(1)) / 2, (lb(2) + ub(2)) / 2, ...
       (lb(3) + ub(3)) / 2, (lb(4) + ub(4)) / 2 ];
if x0(3) <= x0(1) + x0(2) + params.min_hole_sep + x0(4)
    x0(3) = x0(1) + x0(2) + params.min_hole_sep + x0(4) + 0.01; 
    x0(3) = min(x0(3), ub(3)); x0(3) = max(x0(3), lb(3)); 
end

disp('USING 최종 계산된 초기점 x0:'); disp(x0);
disp('USING 최종 계산된 하한 lb:'); disp(lb);
disp('USING 최종 계산된 상한 ub:'); disp(ub);

% =========================================================================
% STAGE 1: Single-Objective Optimization using CUSTOM L-BFGS
% =========================================================================
disp('--- Stage 1: Single-Objective Optimization with CUSTOM L-BFGS ---');

% Options for custom_lbfgs_with_penalty
options_custom_lbfgs = struct();
options_custom_lbfgs.MaxIterations = 200; % Max L-BFGS iterations PER penalty update
options_custom_lbfgs.GradientTolerance = 1e-4; % For augmented objective gradient
options_custom_lbfgs.StepTolerance = 1e-7;
options_custom_lbfgs.FunctionTolerance = 1e-7;
options_custom_lbfgs.HistorySize = 10;      % L-BFGS history (m)
options_custom_lbfgs.FiniteDifferenceStepSize = 1e-7; % For numerical gradient
options_custom_lbfgs.InitialPenalty = 100;    % Starting penalty parameter (mu) -> may need tuning
options_custom_lbfgs.PenaltyFactor = 10;      % Factor to increase mu by (beta)
options_custom_lbfgs.MaxPenaltyIterations = 10;% Max outer penalty update iterations
options_custom_lbfgs.ConstraintTolerancePen = 1e-3; % Target for max_constraint_violation

% --- 1a: Minimize Mass (using Custom L-BFGS) ---
disp(['Optimizing for Minimum Mass (CUSTOM L-BFGS) (must support F_ext >= ', num2str(params.F_ext_min_for_mass_opt), ' N)...']);
% Original objective function (takes x_reduced, params)
obj_mass_for_custom = @(x_r) objective_mass_v3(x_r, params); 
% Nonlinear constraint function (takes x_reduced, params, returns [c, ceq])
nonlcon_min_mass_for_custom = @(x_r) nonlcon_min_mass_v3(x_r, params);

% Call custom L-BFGS
[x_opt_mass, mass_min, exitflag_mass, output_mass] = custom_lbfgs_with_penalty(...
    obj_mass_for_custom, x0, lb, ub, nonlcon_min_mass_for_custom, options_custom_lbfgs);

if exitflag_mass > 0 % Or a specific positive flag from custom solver indicating success
    disp('Custom L-BFGS (Min Mass): Converged (possibly to a feasible point).');
    fprintf('Minimum Mass Found: %.4f kg\n', mass_min);
    disp('Design Variables (Min Mass) [L1,R1,L2,R2]:'); disp(x_opt_mass);
    [~, Fext_at_min_mass] = calculate_beam_performance_v3(x_opt_mass, params);
    fprintf('F_ext at this Min Mass design: %.2f N (Constraint: >= %.2f N)\n', Fext_at_min_mass, params.F_ext_min_for_mass_opt);
    fprintf('Max constraint violation for this solution: %e\n', output_mass.maxconstraintviolation);
    visualize_beam_schematic_v3(x_opt_mass, params, 'beam_min_mass_custom_lbfgs_result.png');
else
    disp('Custom L-BFGS (Min Mass): Did Not Converge or Found No Feasible Solution.');
    fprintf('Exit flag: %d, Message: %s\n', exitflag_mass, output_mass.message);
    fprintf('Max constraint violation for this solution: %e\n', output_mass.maxconstraintviolation);
    if ~isempty(x_opt_mass), visualize_beam_schematic_v3(x_opt_mass, params, 'beam_min_mass_custom_lbfgs_FAIL.png'); end
end
pause(0.5); 

% --- 1b: Maximize F_ext (using Custom L-BFGS) ---
disp(['Optimizing for Maximum F_ext (CUSTOM L-BFGS) (Mass <= ', num2str(params.M_max_for_Fext_opt), ' kg)...']);
obj_neg_Fext_for_custom = @(x_r) objective_neg_Fext_v3(x_r, params);
nonlcon_max_Fext_for_custom = @(x_r) nonlcon_max_Fext_v3(x_r, params);

x0_for_Fext = x0; 
if exitflag_mass > 0 && ~isempty(x_opt_mass) && output_mass.maxconstraintviolation < options_custom_lbfgs.ConstraintTolerancePen * 10
   x0_for_Fext = x_opt_mass; % Use previous good result if available
end

[x_opt_Fext, neg_Fext_max, exitflag_Fext, output_Fext] = custom_lbfgs_with_penalty(...
    obj_neg_Fext_for_custom, x0_for_Fext, lb, ub, nonlcon_max_Fext_for_custom, options_custom_lbfgs);

if exitflag_Fext > 0
    disp('Custom L-BFGS (Max F_ext): Converged (possibly to a feasible point).');
    Fext_max = -neg_Fext_max;
    fprintf('Maximum F_ext Found: %.2f N\n', Fext_max);
    disp('Design Variables (Max F_ext) [L1,R1,L2,R2]:'); disp(x_opt_Fext);
    mass_at_max_Fext_val_array = calculate_beam_performance_v3(x_opt_Fext, params); 
    fprintf('Mass at this Max F_ext design: %.4f kg (Constraint: <= %.4f kg)\n', mass_at_max_Fext_val_array(1), params.M_max_for_Fext_opt);
    fprintf('Max constraint violation for this solution: %e\n', output_Fext.maxconstraintviolation);
    visualize_beam_schematic_v3(x_opt_Fext, params, 'beam_max_Fext_custom_lbfgs_result.png');
else
    disp('Custom L-BFGS (Max F_ext): Did Not Converge or Found No Feasible Solution.');
    fprintf('Exit flag: %d, Message: %s\n', exitflag_Fext, output_Fext.message);
    fprintf('Max constraint violation for this solution: %e\n', output_Fext.maxconstraintviolation);
    if ~isempty(x_opt_Fext), visualize_beam_schematic_v3(x_opt_Fext, params, 'beam_max_Fext_custom_lbfgs_FAIL.png'); end
end
pause(0.5); 

% =========================================================================
% STAGE 2: Multi-Objective Optimization using gamultiobj (unchanged for now)
% =========================================================================
disp('--- Stage 2: Multi-Objective Optimization with gamultiobj (using built-in constraints) ---');
% Note: gamultiobj will use its own robust constraint handling.
% For a "fairer" comparison if you wanted to test custom L-BFGS within a GA framework,
% that would be an even more advanced task (e.g. L-BFGS as a local search operator in GA).

obj_multi_handle_v3 = @(x) objective_multi_v3(x, params);
nonlcon_geom_only_handle = @(x) nonlcon_geom_only_v3(x, params); 

options_ga = optimoptions('gamultiobj', ...
    'PopulationSize', 200, 'CrossoverFraction', 0.8, 'MutationFcn', @mutationadaptfeasible, ...
    'SelectionFcn', @selectiontournament, 'ParetoFraction', 0.4, 'MaxGenerations', 150 * params.nvars, ...
    'FunctionTolerance', 1e-4, 'ConstraintTolerance', 1e-6, ...
    'PlotFcn', {@gaplotpareto, @gaplotscorediversity}, 'Display', 'iter', 'UseParallel', false);

[x_pareto, fval_pareto, exitflag_ga, output_ga] = gamultiobj(obj_multi_handle_v3, params.nvars, ...
    [], [], [], [], lb, ub, nonlcon_geom_only_handle, options_ga);

if exitflag_ga > 0 || exitflag_ga == -4 
    disp('Multi-Objective Optimization (gamultiobj) Finished.');
    pareto_mass = fval_pareto(:,1); pareto_Fext = -fval_pareto(:,2);
    figure('Name', 'Final Pareto Front (V4 - GA)');
    scatter(pareto_mass, pareto_Fext, 50, 'b', 'filled', 'MarkerFaceAlpha', 0.7);
    xlabel('Beam Mass (kg)'); ylabel('Maximum Allowable External Load F_{ext} (N)');
    title(['Pareto (gamultiobj) (L_{total}=', num2str(params.L_total_fixed), 'm, B=', num2str(params.B_fixed), 'm)']);
    grid on; saveas(gcf, 'final_pareto_front_v4_ga.png');
    disp('Final Pareto front plot (gamultiobj) saved as final_pareto_front_v4_ga.png');

    if ~isempty(x_pareto) 
        [x_knee, f_knee] = select_knee_point_v3(x_pareto, fval_pareto);
        if ~isempty(x_knee)
            fprintf('\nKnee Point Selected (from gamultiobj):\n');
            fprintf('  Mass: %.4f kg, F_ext: %.2f N\n', f_knee(1), -f_knee(2));
            disp(x_knee);
            visualize_beam_schematic_v3(x_knee, params, 'beam_visualization_knee_point_v4_ga.png');
            hold on;
            scatter(f_knee(1), -f_knee(2), 150, 'r', 'filled', 'MarkerEdgeColor','k', 'DisplayName','Knee Point');
            legend('Pareto Solutions', 'Knee Point', 'Location', 'Best'); hold off;
            saveas(gcf, 'final_pareto_front_with_knee_v4_ga.png');
        end
    else disp('No Pareto solutions from gamultiobj.'); end
else disp('Multi-Objective Optimization (gamultiobj) Did Not Converge Well.'); end

disp('Optimization Run Finished (Version 4).');
end % END OF MAIN FUNCTION


% --- Helper Functions (Objective Wrappers, Performance Calc, Constraints, Knee Point, Visualization) ---
% --- These are assumed to be the same as in run_beam_optimization_v3.m ---
% --- Ensure objective_mass_v3, objective_neg_Fext_v3, objective_multi_v3, ---
% --- calculate_beam_performance_v3, nonlcon_geom_only_v3, nonlcon_min_mass_v3, ---
% --- nonlcon_max_Fext_v3, select_knee_point_v3, visualize_beam_schematic_v3 ---
% --- are defined below this point or accessible on MATLAB path. ---
% --- For brevity, I am not repeating them here, but they MUST be present in the same file ---
% --- OR in separate files on the MATLAB path. ---
% --- The provided custom_lbfgs_with_penalty.m should be a separate file. ---

% [PASTE THE HELPER FUNCTIONS FROM run_beam_optimization_v3.m HERE]
% Specifically:
% try_save_fmincon_plots (though it's less relevant for custom L-BFGS)
% objective_mass_v3
% objective_neg_Fext_v3
% objective_multi_v3
% calculate_beam_performance_v3 (The one with if/else, not ternary)
% nonlcon_geom_only_v3
% nonlcon_min_mass_v3
% nonlcon_max_Fext_v3
% select_knee_point_v3
% visualize_beam_schematic_v3
% [END OF PASTE SECTION]
% =========================================================================
% HELPER FUNCTIONS (Version 3.1 - with ternary operator fix)
% =========================================================================

function try_save_fmincon_plots(prefix)
    try
        fig_handles = findall(0, 'Type', 'Figure');
        plot_names_to_save = {'Current Function Value', 'Maximum Constraint Violation', 'Current Step Size'};
        plot_names_matlab_internal = {'optimplotfval', 'optimplotconstrviolation', 'optimplotstepsize'}; 

        for i = 1:length(fig_handles)
            fig_name = get(fig_handles(i), 'Name');
            fig_tag = get(fig_handles(i), 'Tag'); 

            found_and_saved = false;
            for j = 1:length(plot_names_to_save)
                if strcmp(fig_name, plot_names_to_save{j}) || strcmp(fig_tag, plot_names_matlab_internal{j})
                    filename = sprintf('fmincon_%s_%s_history.png', prefix, lower(strrep(plot_names_to_save{j}, ' ', '_')));
                    
                    % Avoid saving GA or other main plots
                    if ~(contains(fig_name, 'Pareto') || contains(fig_name, 'Beam Visualization') || contains(fig_name,'Diversity'))
                        saveas(fig_handles(i), filename);
                        disp(['Saved fmincon plot: ', filename]);
                        % close(fig_handles(i)); % Optionally close after saving to reduce clutter
                    end
                    found_and_saved = true;
                    break; 
                end
            end
        end
    catch ME
        disp(['Warning: Could not save fmincon iteration plots for ', prefix, '. Error: ', ME.message]);
    end
end

function mass = objective_mass_v3(x_reduced, params)
    [mass, ~] = calculate_beam_performance_v3(x_reduced, params);
end

function neg_Fext = objective_neg_Fext_v3(x_reduced, params)
    [~, Fext] = calculate_beam_performance_v3(x_reduced, params);
    neg_Fext = -Fext;
end

function f_multi = objective_multi_v3(x_reduced, params)
    [mass, Fext] = calculate_beam_performance_v3(x_reduced, params);
    f_multi = [mass, -Fext];
end

% --- Core Engineering Model: Calculate Mass and F_ext (Version 3.1 - Ternary Op Fixed) ---
function [mass, F_ext_final, varargout] = calculate_beam_performance_v3(x_reduced, params)
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
    F_max_stress_h1 = 0; 
    L_eff1 = L_total - L1; B_net1 = B - 2*R1;
    if B_net1 > 1e-6 && W_fixed > 1e-6 
        S_net1 = (W_fixed * B_net1^2) / 6; Kt1 = Kt_func(R1, B);
        M_sw_h1 = (w_self * L_eff1^2) / 2;
        
        sigma_sw_nom_h1 = inf; 
        if S_net1 > 1e-12
            sigma_sw_nom_h1 = M_sw_h1 / S_net1;
        end

        if L_eff1 <= 1e-6 
            if Kt1 > 1e-6 && (sigma_limit / Kt1) <= sigma_sw_nom_h1 
                F_max_stress_h1 = 0; 
            else
                F_max_stress_h1 = inf; 
            end
        elseif Kt1 > 1e-6 && (sigma_limit / Kt1) > sigma_sw_nom_h1 
            F_max_stress_h1 = ((sigma_limit / Kt1 - sigma_sw_nom_h1) * S_net1) / L_eff1;
        else 
            F_max_stress_h1 = 0; 
        end
    end
    F_max_stress_h1 = max(0, F_max_stress_h1); 
    
    % Hole 2
    F_max_stress_h2 = 0; 
    L_eff2 = L_total - L2; B_net2 = B - 2*R2;
    if B_net2 > 1e-6 && W_fixed > 1e-6 
        S_net2 = (W_fixed * B_net2^2) / 6; Kt2 = Kt_func(R2, B);
        M_sw_h2 = (w_self * L_eff2^2) / 2;

        sigma_sw_nom_h2 = inf; 
        if S_net2 > 1e-12
            sigma_sw_nom_h2 = M_sw_h2 / S_net2;
        end
        
        if L_eff2 <= 1e-6 
            if Kt2 > 1e-6 && (sigma_limit / Kt2) <= sigma_sw_nom_h2 
                F_max_stress_h2 = 0; 
            else
                F_max_stress_h2 = inf; 
            end
        elseif Kt2 > 1e-6 && (sigma_limit / Kt2) > sigma_sw_nom_h2 
            F_max_stress_h2 = ((sigma_limit / Kt2 - sigma_sw_nom_h2) * S_net2) / L_eff2;
        else 
            F_max_stress_h2 = 0; 
        end
    end
    F_max_stress_h2 = max(0, F_max_stress_h2); 

    F_max_stress = min([F_max_stress_fixed, F_max_stress_h1, F_max_stress_h2]);
    F_ext_final = min(F_max_deform, F_max_stress);
    F_ext_final = max(0, F_ext_final); 

    if nargout > 2
        varargout{1} = F_max_deform; varargout{2} = F_max_stress;
    end
end


function [c_geom, ceq_geom] = nonlcon_geom_only_v3(x_reduced, params)
    L1 = x_reduced(1); R1 = x_reduced(2); L2 = x_reduced(3); R2 = x_reduced(4);
    L_total = params.L_total_fixed; B = params.B_fixed;
    min_end_dist = params.min_end_dist; min_hole_sep = params.min_hole_sep;
    min_ligament = params.min_ligament; pdf_tol = params.tol;
    
    c_geom = zeros(7,1); 
    c_geom(1) = min_end_dist - (L1 - R1) + pdf_tol;                     % Hole 1 from fixed end
    c_geom(2) = min_hole_sep - ((L2 - R2) - (L1 + R1)) + pdf_tol;       % Separation between holes
    c_geom(3) = min_end_dist - (L_total - (L2 + R2)) + pdf_tol;         % Hole 2 from free end
    c_geom(4) = min_ligament - (B/2 - R1) + pdf_tol;                     % Ligament hole 1
    c_geom(5) = min_ligament - (B/2 - R2) + pdf_tol;                     % Ligament hole 2
    
    % Ensure hole 1 is fully within beam (considering its end)
    c_geom(6) = (L1 + R1) - (L_total - min_end_dist) + pdf_tol;         % L1_end <= L_total - min_end_dist
                                                                       % Note: This formulation with +pdf_tol makes it L1+R1 >= L_total-min_end_dist+pdf_tol
                                                                       % Correct should be: (L1+R1) - (L_total - min_end_dist) <= 0 (if no pdf_tol from problem statement)
                                                                       % Adhering to problem statement's c(x)+tol <= 0 implies original constraint was stricter
                                                                       % For this type of constraint (X <= Y  --> X - Y <= 0), the PDF formula would be (X-Y)+tol <= 0
                                                                       % So: (L1+R1) - (L_total - min_end_dist) + pdf_tol <= 0
    
    % Ensure L1 is before L2 (start of hole 2 is after end of hole 1 + separation)
    c_geom(7) = (L1 + R1 + min_hole_sep) - (L2-R2) + pdf_tol;           % (L1_end + sep) <= L2_start
                                                                       % (L1+R1+min_hole_sep) - (L2-R2) + pdf_tol <= 0
    ceq_geom = [];
end

function [c_total, ceq_total] = nonlcon_min_mass_v3(x_reduced, params)
    [c_geom, ceq_geom] = nonlcon_geom_only_v3(x_reduced, params); 
    [~, Fext] = calculate_beam_performance_v3(x_reduced, params); 
    c_performance = params.F_ext_min_for_mass_opt - Fext; % F_ext_min <= Fext --> F_ext_min - Fext <=0
    c_total = [c_geom; c_performance];
    ceq_total = ceq_geom;
end

function [c_total, ceq_total] = nonlcon_max_Fext_v3(x_reduced, params)
    [c_geom, ceq_geom] = nonlcon_geom_only_v3(x_reduced, params); 
    [mass, ~] = calculate_beam_performance_v3(x_reduced, params); 
    c_performance = mass - params.M_max_for_Fext_opt; % Mass <= M_max --> Mass - M_max <=0
    c_total = [c_geom; c_performance];
    ceq_total = ceq_geom;
end

function [x_knee, f_knee] = select_knee_point_v3(x_pareto, fval_pareto)
    f1_vals = fval_pareto(:,1); f2_vals = fval_pareto(:,2);
    if isempty(f1_vals) || isempty(f2_vals), x_knee = []; f_knee = []; warning('select_knee_point: Empty Pareto values.'); return; end
    range_f1 = max(f1_vals) - min(f1_vals); range_f2 = max(f2_vals) - min(f2_vals);
    if range_f1 < eps && range_f2 < eps, x_knee = x_pareto(1,:); f_knee = fval_pareto(1,:); warning('select_knee_point: All Pareto points nearly identical.'); return; end
    
    if range_f1 < eps, f1_norm = zeros(size(f1_vals)); else f1_norm = (f1_vals - min(f1_vals)) / range_f1; end
    if range_f2 < eps, f2_norm = zeros(size(f2_vals)); else f2_norm = (f2_vals - min(f2_vals)) / range_f2; end
        
    distances = sqrt(f1_norm.^2 + f2_norm.^2);
    if isempty(distances), x_knee = x_pareto(1,:); f_knee = fval_pareto(1,:); warning('select_knee_point: Could not calculate distances.'); return; end
    [~, min_idx] = min(distances);
    
    x_knee = x_pareto(min_idx,:); f_knee = fval_pareto(min_idx,:);
end

function visualize_beam_schematic_v3(x_reduced_design, params, output_filename)
    if isempty(x_reduced_design), disp('visualize_beam_schematic_v3: No design variables.'); return; end
    L1 = x_reduced_design(1); R1 = x_reduced_design(2); L2 = x_reduced_design(3); R2 = x_reduced_design(4);
    L_total = params.L_total_fixed; B_beam = params.B_fixed;

    fig_beam = figure('Name', ['Beam Visualization: ', output_filename]); clf; 
    set(fig_beam, 'Visible', 'on'); % Ensure figure is visible for saving
    hold on; axis equal;
    rectangle('Position', [0, -B_beam/2, L_total, B_beam], 'EdgeColor', 'k', 'LineWidth', 1.5, 'FaceColor', [0.9 0.9 0.9]);
    if R1 > 1e-6, rectangle('Position', [L1-R1, -R1, 2*R1, 2*R1], 'Curvature', [1,1], 'EdgeColor', 'b', 'FaceColor', 'w'); plot(L1, 0, 'b+', 'MarkerSize', 8); end
    if R2 > 1e-6, rectangle('Position', [L2-R2, -R2, 2*R2, 2*R2], 'Curvature', [1,1], 'EdgeColor', 'r', 'FaceColor', 'w'); plot(L2, 0, 'r+', 'MarkerSize', 8); end
    plot([0 0], [-B_beam/2*1.1 B_beam/2*1.1], 'k-', 'LineWidth', 4);
    xlabel(['X (m) - FIXED L_{total}: ', num2str(L_total,'%.3f'), ' m']);
    ylabel(['Y (m) - FIXED B: ', num2str(B_beam,'%.3f'), ' m']);
    title_str = sprintf('Beam Design (Top View) - W_{fixed}=%.0fmm (%s)', params.W_fixed*1000, output_filename);
    title(strrep(title_str,'_','\_')); % Escape underscores for title
    grid on;
    if R1 > 1e-6, text(L1, 0.01*B_beam/0.1, [' H1(L=' num2str(L1,'%.2f') ',R=' num2str(R1,'%.3f') ')'], 'Color', 'b','Clipping','on','FontSize',8); end
    if R2 > 1e-6, text(L2, -0.01*B_beam/0.1, [' H2(L=' num2str(L2,'%.2f') ',R=' num2str(R2,'%.3f') ')'], 'Color', 'r','Clipping','on','VerticalAlignment','top','FontSize',8); end
    padding_x = 0.05 * L_total; padding_y = 0.25 * B_beam;
    axis_limits = [-padding_x, L_total + padding_x, -B_beam/2 - padding_y, B_beam/2 + padding_y];
    if any(isinf(axis_limits)) || any(isnan(axis_limits)), disp('Warning: Invalid axis limits for visualization.'); else axis(axis_limits); end
    hold off;
    if nargin > 2 && ~isempty(output_filename)
        try 
            saveas(fig_beam, output_filename); 
            disp(['Beam schematic saved as: ' output_filename]);
        catch ME
            disp(['Warning: Could not save beam schematic (V3): ', ME.message]); 
        end
    end
end