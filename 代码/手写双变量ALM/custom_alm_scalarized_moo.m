function [x_alm_opt, f_scalar_opt, f_objectives_opt, exitflag_alm, output_alm] = ...
    custom_alm_scalarized_moo(scalarized_obj_fun, obj_fun1_handle, obj_fun2_handle, ...
                              x0, lb, ub, nonlcon_geom_fun, options_alm, params_beam)
% CUSTOM_ALM_SCALARIZED_MOO Augmented Lagrangian for scalarized multi-objective.
% scalarized_obj_fun: @(x) w*f1(x) + (1-w)*f2(x)
% obj_fun1_handle, obj_fun2_handle: handles to original f1, f2 (for final reporting)
% x0, lb, ub: Initial guess and bounds.
% nonlcon_geom_fun: @(x, params_beam) that returns [c, ceq] for geometric constraints.
% options_alm: Struct with ALM and inner L-BFGS options.
% params_beam: The 'params' struct needed by nonlcon_geom_fun.

    lambda = options_alm.InitialLambda; 
    mu = options_alm.InitialMu;
    beta = options_alm.PenaltyFactorBeta;
    max_alm_iter = options_alm.MaxAlmIterations;
    constr_tol_alm = options_alm.ConstraintToleranceAlm;
    lbfgs_opts = options_alm.LBFGS_options;

    x_k = x0(:);
    lb = lb(:); ub = ub(:);

    fprintf('--- Starting Custom ALM for Scalarized Objective ---\n');
    
    output_alm.iterations = 0;
    output_alm.all_x = []; 
    output_alm.prev_max_viol = inf; % Initialize for first comparison
    exitflag_alm = -1; % Default if not converged

    for alm_iter = 1:max_alm_iter
        output_alm.iterations = alm_iter;
        output_alm.all_x = [output_alm.all_x, x_k]; % Store trajectory

        fprintf('ALM Iter: %d, Mu: %e, Lambda_norm: %e\n', alm_iter, mu, norm(lambda));
        
        augmented_lagrangian_fun = @(x_inner) ...
            scalarized_obj_fun(x_inner) + ...
            alm_penalty_term(x_inner, lambda, mu, nonlcon_geom_fun, params_beam);
        
        grad_al_numerical_fun = @(x_inner) ...
            numerical_gradient_lbfgs(augmented_lagrangian_fun, x_inner, lbfgs_opts.FiniteDifferenceStepSize); % This will now call the separate .m file

        [x_k_plus_1, ~, exitflag_lbfgs, output_lbfgs_inner] = ...
            custom_lbfgs_box_constrained(augmented_lagrangian_fun, grad_al_numerical_fun, ...
                                         x_k, lb, ub, lbfgs_opts);
        
        if exitflag_lbfgs <= 0 
            fprintf('  ALM: Inner L-BFGS failed or did not fully converge (flag %d, msg: %s). Using result x_k_plus_1.\n', exitflag_lbfgs, output_lbfgs_inner.message);
            if any(isnan(x_k_plus_1)) || any(isinf(x_k_plus_1))
                fprintf('  ALM: x_k_plus_1 from L-BFGS contains NaN/Inf. Reverting to previous x_k for this ALM step.\n');
                x_k_plus_1 = x_k; % Critical: prevent NaN propagation
                if alm_iter == 1 % If fails on first iter, ALM cannot proceed well
                    exitflag_alm = -2; % Indicate ALM failure due to initial LBFGS failure
                    output_alm.message = 'ALM failed: Inner L-BFGS failed critically on first iteration.';
                    break; 
                end
            end
        end
        
        x_k_prev_alm = x_k;
        x_k = x_k_plus_1; 

        [c_geom_k, ~] = nonlcon_geom_fun(x_k, params_beam); 
        if any(isnan(c_geom_k)) || any(isinf(c_geom_k))
            fprintf('  ALM: Constraints c_geom_k are NaN/Inf. Problem with x_k. Stopping ALM.\n');
            exitflag_alm = -5; % Error in constraint evaluation
            output_alm.message = 'ALM failed: Constraint evaluation resulted in NaN/Inf.';
            break;
        end


        lambda_new = max(0, lambda + mu * c_geom_k);
        
        max_constr_viol_k = max(0, max(c_geom_k, [], 'omitnan'));
        if isempty(max_constr_viol_k), max_constr_viol_k = 0; end
        
        fprintf('  ALM Post-LBFGS: MaxGeomViol: %e, LambdaChangeNorm: %e, XChangeNorm: %e\n', ...
                max_constr_viol_k, norm(lambda_new - lambda), norm(x_k - x_k_prev_alm));

        if max_constr_viol_k < constr_tol_alm && ...
           norm(lambda_new - lambda) < constr_tol_alm * sqrt(length(lambda) + eps) && ...
           norm(x_k - x_k_prev_alm) < lbfgs_opts.StepTolerance * 100 % Relaxed x change for ALM convergence
            fprintf('ALM Converged: Constraints and multipliers stabilized.\n');
            exitflag_alm = 1;
            break;
        end
        
        lambda = lambda_new; 

        if alm_iter > 1 
            if max_constr_viol_k > 0.75 * output_alm.prev_max_viol + constr_tol_alm * 0.1 % Add small tolerance to avoid premature mu increase
                 mu = mu * beta;
                 fprintf('  ALM: Increasing mu to %e due to insufficient constraint improvement.\n', mu);
            end
        end
        output_alm.prev_max_viol = max_constr_viol_k;

        if mu > 1e12 % Increased safety break for mu
            fprintf('ALM Warning: Mu is very large (%e). Stopping.\n', mu);
            exitflag_alm = -3; 
            break;
        end
    end 

    if alm_iter == max_alm_iter && exitflag_alm ~= 1 % Check if max iter reached without prior convergence
        fprintf('ALM Warning: Maximum ALM iterations reached.\n');
        exitflag_alm = 0;
    end
    if exitflag_alm ~=1 && max_constr_viol_k > constr_tol_alm
         warning('ALM did not fully satisfy constraint tolerance. Max violation: %e', max_constr_viol_k);
    end

    x_alm_opt = x_k;
    f_scalar_opt = scalarized_obj_fun(x_alm_opt);
    
    try
        f_obj1_val = obj_fun1_handle(x_alm_opt, params_beam); 
        f_obj2_val = obj_fun2_handle(x_alm_opt, params_beam); 
        f_objectives_opt = [f_obj1_val, f_obj2_val];
    catch ME_final_obj
        fprintf('Error evaluating final objectives at x_alm_opt: %s\n', ME_final_obj.message);
        f_objectives_opt = [NaN, NaN];
    end
    
    output_alm.lambda = lambda;
    output_alm.mu = mu;
    output_alm.max_geom_violation = max_constr_viol_k;
    if ~isfield(output_alm, 'message') && exitflag_alm == 0
        output_alm.message = 'ALM: Maximum iterations reached.';
    elseif ~isfield(output_alm, 'message') && exitflag_alm == 1
        output_alm.message = 'ALM: Converged.';
    elseif ~isfield(output_alm, 'message')
        output_alm.message = 'ALM: Exited with unspecified condition.';
    end
end

% --- Helper: ALM Penalty Term for c(x) <= 0 ---
function penalty_val = alm_penalty_term(x, lambda, mu, nonlcon_geom_fun, params_beam)
    [c_geom, ~] = nonlcon_geom_fun(x, params_beam); 
    
    term_sum = 0;
    if ~isempty(c_geom) && ~any(isnan(c_geom)) && ~any(isinf(c_geom)) % Check for valid c_geom
        for i = 1:length(c_geom)
            term_sum = term_sum + (max(0, lambda(i) + mu * c_geom(i)))^2 - lambda(i)^2;
        end
    else
        penalty_val = 1e20; % Large penalty if constraints are invalid
        warning('alm_penalty_term: c_geom contains NaN/Inf or is empty. Applying large penalty.');
        return;
    end
    penalty_val = (1 / (2 * mu)) * term_sum;
end