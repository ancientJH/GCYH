function [x_opt, f_opt, exitflag, output] = custom_lbfgs_with_penalty(obj_fun_original, x0, lb, ub, nonlcon_fun, options_custom)
% CUSTOM_LBFGS_WITH_PENALTY A simplified L-BFGS with quadratic penalty for constraints.
%
% Inputs:
%   obj_fun_original: Handle to the original objective function f(x).
%   x0: Initial guess.
%   lb, ub: Lower and upper bounds on x.
%   nonlcon_fun: Handle to the nonlinear constraint function [c, ceq] = nonlcon(x).
%                c(x) <= 0, ceq(x) = 0.
%   options_custom: Struct with custom L-BFGS and penalty options.
%       .MaxIterations: Max iterations for inner L-BFGS loop.
%       .GradientTolerance: Tolerance for gradient norm.
%       .StepTolerance: Tolerance for step size.
%       .FunctionTolerance: Tolerance for function value change.
%       .HistorySize (m): L-BFGS history size.
%       .FiniteDifferenceStepSize: Step for numerical gradient.
%       .InitialPenalty (mu): Starting penalty parameter.
%       .PenaltyFactor (beta): Factor to increase penalty by (e.g., 10).
%       .MaxPenaltyIterations: Max outer loop iterations for updating penalty.
%       .ConstraintTolerancePen: Target constraint satisfaction for penalty method.
%
% Outputs:
%   x_opt: Optimal solution found.
%   f_opt: Original objective function value at x_opt.
%   exitflag: Exit condition.
%   output: Struct with optimization information.

    % --- Unpack Options ---
    max_iter_lbfgs = options_custom.MaxIterations;
    grad_tol = options_custom.GradientTolerance;
    step_tol = options_custom.StepTolerance;
    func_tol = options_custom.FunctionTolerance;
    m_history = options_custom.HistorySize;
    h_diff = options_custom.FiniteDifferenceStepSize;
    
    mu = options_custom.InitialPenalty;
    beta_penalty = options_custom.PenaltyFactor;
    max_penalty_iter = options_custom.MaxPenaltyIterations;
    constr_tol_penalty = options_custom.ConstraintTolerancePen;

    n = length(x0);
    x = x0(:); % Ensure column vector
    lb = lb(:); ub = ub(:);

    % L-BFGS history (s_k = x_{k+1}-x_k, y_k = grad_L(x_{k+1})-grad_L(x_k))
    s_hist = zeros(n, 0); 
    y_hist = zeros(n, 0);

    iter_total = 0;
    converged_overall = false;
    
    fprintf('\n--- Starting Custom L-BFGS with Penalty Method ---\n');

    % --- Outer Loop: Penalty Parameter Update ---
    for penalty_iter = 1:max_penalty_iter
        fprintf('Penalty Iteration: %d, Penalty Parameter (mu): %e\n', penalty_iter, mu);
        
        x_penalty_iter_start = x; % Store x at the start of this penalty iteration

        % --- Inner Loop: L-BFGS for current augmented Lagrangian ---
        for k = 0:max_iter_lbfgs-1
            iter_total = iter_total + 1;

            % Augmented Objective Function P(x, mu)
            % P(x,mu) = f(x) + (mu/2) * ( sum(max(0,c_i(x))^2) + sum(ceq_j(x)^2) + sum(max(0,lb-x)^2) + sum(max(0,x-ub)^2) )
            [P_x, f_x, c_x, ceq_x] = augmented_objective(x, obj_fun_original, nonlcon_fun, lb, ub, mu);
            
            % Numerical Gradient of P(x, mu)
            grad_P_x = numerical_gradient(@(xt) augmented_objective(xt, obj_fun_original, nonlcon_fun, lb, ub, mu), x, h_diff);

            if norm(grad_P_x) < grad_tol
                fprintf('  L-BFGS inner loop: Gradient norm below tolerance (%e < %e).\n', norm(grad_P_x), grad_tol);
                break; % Converged for this mu
            end

            % L-BFGS Search Direction Calculation (Two-Loop Recursion)
            q = grad_P_x;
            num_corr = size(s_hist, 2);
            alpha_corr = zeros(num_corr, 1);

            % First loop (alpha)
            for i = num_corr:-1:1
                rho_i = 1 / (s_hist(:,i)' * y_hist(:,i) + eps); % eps for stability
                alpha_corr(i) = rho_i * (s_hist(:,i)' * q);
                q = q - alpha_corr(i) * y_hist(:,i);
            end

            % Initial Hessian approximation H0 (scaled identity)
            if k > 0 && num_corr > 0
                gamma_k = (s_hist(:,end)' * y_hist(:,end)) / (y_hist(:,end)' * y_hist(:,end) + eps);
                H0_diag_val = gamma_k;
            else
                H0_diag_val = 1 / max(1, norm(grad_P_x)); % Heuristic scaling for first step
            end
            r = H0_diag_val * q;

            % Second loop (beta)
            for i = 1:num_corr
                rho_i = 1 / (s_hist(:,i)' * y_hist(:,i) + eps);
                beta_i = rho_i * (y_hist(:,i)' * r);
                r = r + s_hist(:,i) * (alpha_corr(i) - beta_i);
            end
            
            pk = -r; % Search direction

            % Line Search (Backtracking with Armijo condition)
            alpha_step = 1.0; % Initial step size
            c1_armijo = 1e-4;
            rho_backtrack = 0.5;
            max_ls_iters = 20;
            x_old_ls = x; % Store current x for s_k update
            
            ls_iter_count = 0;
            for ls_iter = 1:max_ls_iters
                ls_iter_count = ls_iter;
                x_new = x_old_ls + alpha_step * pk;
                % No explicit projection here; bounds are handled by penalty.
                % Could add projection if desired for L-BFGS-B style.
                
                P_x_new = augmented_objective(x_new, obj_fun_original, nonlcon_fun, lb, ub, mu);
                
                if P_x_new <= P_x + c1_armijo * alpha_step * (grad_P_x' * pk) % Armijo condition
                    break; 
                end
                alpha_step = rho_backtrack * alpha_step;
                if alpha_step < 1e-12 % Step too small
                    break;
                end
            end
            
            if ls_iter_count == max_ls_iters && P_x_new > P_x + c1_armijo * alpha_step * (grad_P_x' * pk)
                fprintf('  L-BFGS inner loop: Line search failed to find suitable step after %d iterations. Gradient norm: %e. Direction norm: %e.\n', max_ls_iters, norm(grad_P_x), norm(pk));
                % Could try a gradient descent step or stop
                pk = -grad_P_x / max(1, norm(grad_P_x)); % Normalized GD step
                alpha_step = 1e-4; % Small GD step
                x_new = x_old_ls + alpha_step * pk;
                P_x_new = augmented_objective(x_new, obj_fun_original, nonlcon_fun, lb, ub, mu);
                 if P_x_new > P_x
                     fprintf('  L-BFGS inner loop: Fallback GD step also did not improve. Stopping inner loop.\n');
                     break; % Stop L-BFGS for this mu
                 end
            end
            x = x_new

            % Update L-BFGS history
            s_k = x - x_old_ls;
            grad_P_x_new = numerical_gradient(@(xt) augmented_objective(xt, obj_fun_original, nonlcon_fun, lb, ub, mu), x, h_diff);
            y_k = grad_P_x_new - grad_P_x;

            if (s_k' * y_k) > 1e-9 % Curvature condition (simplified)
                if size(s_hist, 2) >= m_history
                    s_hist = s_hist(:, 2:end);
                    y_hist = y_hist(:, 2:end);
                end
                s_hist = [s_hist, s_k];
                y_hist = [y_hist, y_k];
            else
                fprintf('  L-BFGS inner loop: Skipping update, y_k^T s_k = %e (not positive enough).\n', s_k'*y_k);
            end
            
            % Termination checks for inner loop
            if norm(s_k) < step_tol * (1 + norm(x_old_ls))
                fprintf('  L-BFGS inner loop: Step size below tolerance after %d L-BFGS iterations.\n', k+1);
                break;
            end
            if abs(P_x_new - P_x) < func_tol * (1 + abs(P_x)) && norm(grad_P_x) < grad_tol * 10 % Relaxed grad_tol check if func_tol met
                fprintf('  L-BFGS inner loop: Function value change below tolerance after %d L-BFGS iterations.\n', k+1);
                break;
            end
            
            if mod(k, 20) == 0
                max_viol_curr = max_constraint_violation(x, nonlcon_fun, lb, ub);
                fprintf('  L-BFGS iter %d: AugObj=%e, OrigObj=%e, GradNorm=%e, MaxViol=%e, Step=%e\n', ...
                        k, P_x_new, obj_fun_original(x), norm(grad_P_x_new), max_viol_curr, alpha_step);
            end
        end % End inner L-BFGS loop
        
        % Check constraint violation after inner L-BFGS loop for current mu
        current_max_violation = max_constraint_violation(x, nonlcon_fun, lb, ub);
        fprintf('Penalty Iteration %d finished. Max constraint violation: %e\n', penalty_iter, current_max_violation);

        if current_max_violation < constr_tol_penalty
            fprintf('Overall convergence: Constraints satisfied within tolerance %e.\n', constr_tol_penalty);
            converged_overall = true;
            break; % Exit outer penalty loop
        end
        
        % Update penalty parameter if not converged on constraints
        mu = mu * beta_penalty;
        
        % Check for stagnation in x across penalty iterations
        if norm(x - x_penalty_iter_start) < step_tol * 10 && penalty_iter > 1
            fprintf('Outer loop: Stagnation in x. Max violation still %e. Stopping.\n', current_max_violation);
            break;
        end

    end % End outer penalty loop

    x_opt = x;
    f_opt = obj_fun_original(x_opt); % Return original objective value
    
    % Set exitflag and output
    output.iterations = iter_total;
    output.penaltyparameter = mu;
    output.maxconstraintviolation = max_constraint_violation(x_opt, nonlcon_fun, lb, ub);
    
    if converged_overall
        exitflag = 1; % Converged to a feasible point
        output.message = 'Converged: Constraints satisfied within tolerance.';
    elseif output.maxconstraintviolation < constr_tol_penalty * 10 % Relaxed feasibility
        exitflag = 2; % Converged to a nearly feasible point
        output.message = 'Converged to a nearly feasible point (outer loop max iter or stagnation).';
         warning('Custom L-BFGS: Final solution has constraint violation %e', output.maxconstraintviolation);
    else
        exitflag = -2; % Did not converge to a feasible point
        output.message = 'Did not converge to a feasible point after max penalty iterations or stagnation.';
        warning('Custom L-BFGS: Failed to converge to a feasible solution. Max violation: %e', output.maxconstraintviolation);
    end
end

% --- Helper: Augmented Objective Function Value ---
function [P_val, f_val, c_val, ceq_val] = augmented_objective(x, obj_fun, nonlcon_fun, lb, ub, mu)
    f_val = obj_fun(x);
    [c_val, ceq_val] = nonlcon_fun(x);

    penalty = 0;
    if ~isempty(c_val)
        penalty = penalty + sum(max(0, c_val).^2);
    end
    if ~isempty(ceq_val)
        penalty = penalty + sum(ceq_val.^2);
    end
    % Bound penalties
    penalty = penalty + sum(max(0, lb-x).^2) + sum(max(0, x-ub).^2);
    
    P_val = f_val + (mu/2) * penalty;
end

% --- Helper: Numerical Gradient (Forward Difference) ---
function grad = numerical_gradient(fun_handle, x, h)
    % fun_handle here should be the one that returns only the scalar P_val
    n = length(x);
    grad = zeros(n, 1);
    P_x = fun_handle(x); % Evaluate P at current x
    for i = 1:n
        x_h = x;
        x_h(i) = x_h(i) + h;
        P_x_h = fun_handle(x_h);
        grad(i) = (P_x_h - P_x) / h;
    end
end

% --- Helper: Max Constraint Violation ---
function max_viol = max_constraint_violation(x, nonlcon_fun, lb, ub)
    [c, ceq] = nonlcon_fun(x);
    viol_c = max(0, max(c, [], 'omitnan')); % Max of positive c_i
    viol_ceq = max(abs(ceq), [], 'omitnan');
    viol_lb = max(0, max(lb-x, [], 'omitnan'));
    viol_ub = max(0, max(x-ub, [], 'omitnan'));
    max_viol = max([0; viol_c; viol_ceq; viol_lb; viol_ub]);
     if isempty(max_viol), max_viol = 0; end % Handle case where all constraints are empty
end