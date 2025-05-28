function run_beam_optimization_v5 % Renamed main function
% =========================================================================
% MAIN SCRIPT for Multi-Objective Cantilever Beam Optimization (Version 5)
% =========================================================================
% - Single-objective uses fmincon (as per V3).
% - Multi-objective uses custom ALM with scalarization and custom L-BFGS.

clear; close all; clc;
disp('Starting Cantilever Beam Optimization (Version 5 - Custom ALM for MOO)...');

% --- Define Fixed Parameters and Material Properties (Same as V3) ---
params.rho = 7850; params.E = 200e9; params.sigma_limit = 250e6;
params.D_limit = 0.0035; params.W_fixed = 0.02; params.g = 9.81;
params.L_total_fixed = 1.2; params.B_fixed = 0.12;
disp(['FIXED Beam Dimensions: L_total = ', num2str(params.L_total_fixed), 'm, B = ', num2str(params.B_fixed), 'm']);
params.F_ext_min_for_mass_opt = 1000; params.M_max_for_Fext_opt = 25.0;
disp(['Constraint for Max F_ext opt: Mass <= ', num2str(params.M_max_for_Fext_opt), ' kg']);
disp(['Constraint for Min Mass opt: F_ext >= ', num2str(params.F_ext_min_for_mass_opt), ' N']);
params.min_end_dist = 0.01; params.min_hole_sep = 0.01;
params.min_ligament = 0.005; params.tol = 1e-6;
params.nvars = 4;

% --- Design Variable Bounds & Initial Guess (Simplified from V3.1 for robustness) ---
R_min_abs = 0.005;
R_max_abs = params.B_fixed/2 - params.min_ligament - params.tol;
if R_max_abs <= R_min_abs, error('R_max_abs too small. Check B_fixed/min_ligament.'); end

L1_min_abs = params.min_end_dist + R_min_abs; % Min L1 if R1 is min
L1_max_abs = params.L_total_fixed - params.min_end_dist - R_min_abs - params.min_hole_sep - 2*R_min_abs; % Max L1 to leave space for hole2
if L1_max_abs <= L1_min_abs, L1_max_abs = L1_min_abs + 0.1*params.L_total_fixed; warning('L1_max_abs adjusted'); end


L2_min_abs = L1_min_abs + 2*R_min_abs + params.min_hole_sep; % Min L2 after a min hole1
L2_max_abs = params.L_total_fixed - params.min_end_dist - R_min_abs; % Max L2 if R2 is min
if L2_max_abs <= L2_min_abs, L2_max_abs = L2_min_abs + 0.1*params.L_total_fixed; warning('L2_max_abs adjusted'); end

lb = [L1_min_abs, R_min_abs, L2_min_abs, R_min_abs];
ub = [L1_max_abs, R_max_abs, L2_max_abs, R_max_abs];

% Ensure lb < ub
for i=1:params.nvars
    if lb(i) >= ub(i)
        lb(i) = ub(i) * 0.9; % Adjust lb if it's not strictly less
        if lb(i) >= ub(i), error('Bound setup error: lb(%d) >= ub(%d) after adjustment.',i,i); end
        warning('Adjusted lb(%d) because it was >= ub(%d). Review bounds.',i,i);
    end
end
x0 = (ub+lb)/2; %initial guess
if x0(3) <= x0(1) + x0(2) + params.min_hole_sep + x0(4)
    x0(3) = x0(1) + x0(2) + params.min_hole_sep + x0(4) + 0.01; 
    x0 = max(lb, min(x0, ub)); 
end

disp('USING 초기점 x0:'); disp(x0);
disp('USING 하한 lb:'); disp(lb);
disp('USING 상한 ub:'); disp(ub);

% =========================================================================
% STAGE 1: Single-Objective Optimization using fmincon (Retained from V3)
% =========================================================================
% ... (fmincon calls for min_mass and max_Fext from run_beam_optimization_v3.m can be pasted here) ...
% ... For brevity, I'm skipping pasting this part. Assume it runs and x_opt_mass, x_opt_Fext are available if needed.
disp('--- Stage 1: Single-Objective Optimization with fmincon (SKIPPED IN THIS EXAMPLE FOR BREVITY, ASSUME RESULTS AVAILABLE IF NEEDED) ---');
% If you need these results as starting points for ALM, run them first.
x_opt_mass_fmincon = x0; % Placeholder
x_opt_Fext_fmincon = x0; % Placeholder


% =========================================================================
% STAGE 2: Multi-Objective Optimization using CUSTOM ALM + L-BFGS
% =========================================================================
disp('--- Stage 2: Multi-Objective Optimization with Custom ALM ---');

% Define original objective function handles (these take x_reduced and params)
obj_fun1_mass_handle = @(x, p) objective_mass_v3(x, p);
obj_fun2_negFext_handle = @(x, p) objective_neg_Fext_v3(x, p); % Returns -Fext

% Define geometric constraint handle (takes x_reduced and params)
nonlcon_geom_handle = @(x, p) nonlcon_geom_only_v3(x, p);

% Setup for ALM and L-BFGS
alm_options = struct();
% Get number of geometric constraints for initial lambda
[c_test, ~] = nonlcon_geom_handle(x0, params);
alm_options.InitialLambda = zeros(length(c_test), 1); % Initialize lambdas to 0
alm_options.InitialMu = 10;         % Initial penalty parameter -> CRITICAL TUNING PARAMETER
alm_options.PenaltyFactorBeta = 5;  % Factor to increase mu -> CRITICAL TUNING PARAMETER
alm_options.MaxAlmIterations = 15;  % Max outer ALM iterations
alm_options.ConstraintToleranceAlm = 1e-3; % Target for max constraint violation in ALM

lbfgs_custom_options = struct();
lbfgs_custom_options.MaxIterations = 100; % Max L-BFGS iterations PER ALM subproblem
lbfgs_custom_options.GradientTolerance = 1e-6; % For augmented Lagrangian gradient
lbfgs_custom_options.StepTolerance = 1e-5;
lbfgs_custom_options.FunctionTolerance = 1e-7;
lbfgs_custom_options.HistorySize = 7;
lbfgs_custom_options.FiniteDifferenceStepSize = 1e-7;
alm_options.LBFGS_options = lbfgs_custom_options;

% Scalarization loop
num_weights = 11; % Number of points to try on Pareto front (e.g., 0, 0.1, ..., 1.0)
weights = linspace(0.05, 0.95, num_weights); % Avoid 0 and 1 to prevent issues if one obj is flat
% weights = [0.1, 0.3, 0.5, 0.7, 0.9]; % Fewer points for faster testing

pareto_points_x = [];
pareto_points_f = [];

x_start_alm = x0; % Initial guess for the first ALM run

for i = 1:length(weights)
    w = weights(i);
    fprintf('\n>>> Solving ALM for weight w = %.2f <<<\n', w);

    % Define the scalarized objective for the current weight w
    % It needs to take only x as input for custom_alm_scalarized_moo's scalarized_obj_fun
    current_scalarized_obj_fun = @(x_inner) ...
        w * obj_fun1_mass_handle(x_inner, params) + ...
        (1-w) * obj_fun2_negFext_handle(x_inner, params);

    % Call the custom ALM solver
    [x_opt_w, ~, f_objectives_at_x_opt_w, exitflag_alm_w, output_alm_w] = ...
        custom_alm_scalarized_moo(current_scalarized_obj_fun, ...
                                  obj_fun1_mass_handle, ... % Pass original f1 handle
                                  @(x_r,p) -obj_fun2_negFext_handle(x_r,p), ... % Pass original f2 handle (so it's F_ext)
                                  x_start_alm, lb, ub, ...
                                  nonlcon_geom_handle, alm_options, params); % Pass params

    if exitflag_alm_w >= 0 % Consider solution if ALM didn't completely fail
        if output_alm_w.max_geom_violation < alm_options.ConstraintToleranceAlm * 10 % Relaxed check for storing point
            pareto_points_x = [pareto_points_x; x_opt_w'];
            % f_objectives_at_x_opt_w is [mass, F_ext] because we passed -obj_fun2_negFext to get F_ext
            pareto_points_f = [pareto_points_f; f_objectives_at_x_opt_w(1), f_objectives_at_x_opt_w(2)];
            fprintf('Stored Pareto point for w=%.2f: Mass=%.3f, Fext=%.2f (Viol: %.2e)\n', ...
                    w, f_objectives_at_x_opt_w(1), f_objectives_at_x_opt_w(2), output_alm_w.max_geom_violation);
            x_start_alm = x_opt_w; % Use this solution as start for next weight (warm start)
        else
            fprintf('Skipped storing point for w=%.2f due to high constraint violation: %.2e\n', w, output_alm_w.max_geom_violation);
        end
    else
        fprintf('ALM failed for w=%.2f (exitflag %d).\n', w, exitflag_alm_w);
    end
end

% Plotting the custom Pareto Front
if ~isempty(pareto_points_f)
    figure('Name', 'Custom ALM Pareto Front (V5)');
    plot(pareto_points_f(:,1), pareto_points_f(:,2), 'ro-', 'MarkerFaceColor', 'r');
    xlabel('Beam Mass (kg)');
    ylabel('Maximum Allowable External Load F_{ext} (N)');
    title(['Custom ALM Pareto Front (L_{total}=', num2str(params.L_total_fixed), 'm, B=', num2str(params.B_fixed), 'm)']);
    grid on;
    saveas(gcf, 'custom_alm_pareto_front_v5.png');
    disp('Custom ALM Pareto front plot saved as custom_alm_pareto_front_v5.png');

    % Select and visualize knee point from custom Pareto front
    % Need to provide fval_pareto format: [obj1, -obj2_to_min]
    fval_for_knee = [pareto_points_f(:,1), -pareto_points_f(:,2)];
    [x_knee_custom, f_knee_custom_neg] = select_knee_point_v3(pareto_points_x, fval_for_knee);
    
    if ~isempty(x_knee_custom)
        fprintf('\nKnee Point Selected (from Custom ALM Pareto):\n');
        fprintf('  Mass: %.4f kg\n', f_knee_custom_neg(1));
        fprintf('  F_ext: %.2f N\n', -f_knee_custom_neg(2));
        fprintf('  Design Variables [L1, R1, L2, R2]:\n');
        disp(x_knee_custom);
        visualize_beam_schematic_v3(x_knee_custom, params, 'beam_visualization_knee_point_custom_alm_v5.png');
        
        hold on;
        plot(f_knee_custom_neg(1), -f_knee_custom_neg(2), 'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'DisplayName', 'Knee Point (Custom ALM)');
        legend('Custom ALM Pareto', 'Knee Point (Custom ALM)', 'Location', 'Best');
        hold off;
        saveas(gcf, 'custom_alm_pareto_front_with_knee_v5.png');
    else
        disp('Could not select a knee point from the custom ALM Pareto front.');
    end
else
    disp('No Pareto points generated by Custom ALM method.');
end

disp('Optimization Run Finished (Version 5).');
end % END OF MAIN FUNCTION


% --- Helper Functions (Objective Wrappers, Performance Calc, Constraints, Knee Point, Visualization) ---
% PASTE THE FOLLOWING FUNCTIONS FROM run_beam_optimization_v3.m HERE:
% objective_mass_v3
% objective_neg_Fext_v3
% objective_multi_v3 (though not directly used by custom ALM, good to keep if comparing)
% calculate_beam_performance_v3 (The one with if/else, not ternary)
% nonlcon_geom_only_v3
% nonlcon_min_mass_v3
% nonlcon_max_Fext_v3
% select_knee_point_v3
% visualize_beam_schematic_v3
% try_save_fmincon_plots (if you keep the fmincon part for single-objective)
% [END OF PASTE SECTION FOR HELPER FUNCTIONS FROM V3]
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