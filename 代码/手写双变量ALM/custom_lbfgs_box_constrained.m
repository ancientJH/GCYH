function [x_opt, f_opt, exitflag, output] = custom_lbfgs_box_constrained(obj_fun, grad_fun_numerical, x0, lb, ub, options)
% CUSTOM_LBFGS_BOX_CONSTRAINED L-BFGS for box-constrained optimization.
% Robust initialization of outputs and more try-catch blocks.

    % Initialize output arguments to default/error values
    x_opt = nan(size(x0)); 
    f_opt = NaN;
    exitflag = -10; % Custom flag for unhandled exit or pre-loop failure
    output = struct('iterations', 0, 'funcCount', 0, 'message', 'L-BFGS did not start or exited prematurely.');

    n = length(x0);
    if n == 0
        output.message = 'Input x0 is empty.';
        warning('custom_lbfgs_box_constrained: Input x0 is empty.');
        x_opt = x0; % Return empty if input is empty
        return; % Exit early
    end

    x = x0(:); 
    lb = lb(:); ub = ub(:);
    x = max(lb, min(x, ub)); % Project initial guess

    % --- Unpack Options ---
    max_iter = options.MaxIterations;
    grad_tol = options.GradientTolerance;
    step_tol = options.StepTolerance;
    func_tol = options.FunctionTolerance;
    m_history = options.HistorySize;

    s_hist = zeros(n, 0); 
    y_hist = zeros(n, 0);
    
    iter_count = 0; % Tracks actual L-BFGS iterations performed

    % Initial objective and gradient evaluation with error handling
    try
        f_old = obj_fun(x);
        output.funcCount = output.funcCount + 1;
        grad_old = grad_fun_numerical(x); 
        % Assuming numerical_gradient_lbfgs does N*2 evaluations for central diff
        % This funcCount is an approximation for gradient calls
    catch ME_initial
        output.message = ['Error during initial objective/gradient evaluation: ', ME_initial.message];
        warning('custom_lbfgs_box_constrained: %s', output.message);
        x_opt = x; % Return current x (projected x0)
        f_opt = NaN; 
        exitflag = -11; % Custom flag for initial eval failure
        return; % Exit early
    end

    % Check if initial point already satisfies convergence
    if norm(grad_old) < grad_tol && iter_count == 0 
        output.message = 'L-BFGS: Initial point satisfies gradient tolerance.';
        exitflag = 1;
        x_opt = x; f_opt = f_old; output.iterations = 0;
        fprintf('  L-BFGS (box) initial: Obj: %e, GradNorm: %e. Converged at x0.\n', f_old, norm(grad_old));
        return;
    end
    
    fprintf('  L-BFGS (box) iter: %d, Obj: %e, GradNorm: %e\n', iter_count, f_old, norm(grad_old));

    % --- Main L-BFGS Loop ---
    for k = 0:max_iter-1
        iter_count = k + 1; 
        output.iterations = iter_count;

        % L-BFGS Search Direction (Two-Loop Recursion)
        q = grad_old;
        num_corr = size(s_hist, 2);
        alpha_corr = zeros(num_corr, 1);

        for i = num_corr:-1:1
            rho_i_val = s_hist(:,i)' * y_hist(:,i);
            if abs(rho_i_val) < eps 
                alpha_corr(i) = 0; 
            else
                rho_i = 1 / rho_i_val;
                alpha_corr(i) = rho_i * (s_hist(:,i)' * q);
                q = q - alpha_corr(i) * y_hist(:,i);
            end
        end

        if k > 0 && num_corr > 0 % k is 0-indexed, so k>0 means at least one LBFGS step taken
            sy = s_hist(:,end)' * y_hist(:,end);
            yy = y_hist(:,end)' * y_hist(:,end);
            if abs(yy) < eps
                gamma_k = 1.0; % Fallback if yy is too small
            else
                gamma_k = sy / yy;
            end
            H0_diag_val = max(eps, gamma_k); % Ensure H0 is positive
        else
            H0_diag_val = 1 / max(1, norm(grad_old)); 
        end
        r_dir = H0_diag_val * q;

        for i = 1:num_corr
            rho_i_val = s_hist(:,i)' * y_hist(:,i);
             if abs(rho_i_val) < eps
                 beta_i = 0; 
             else
                rho_i = 1 / rho_i_val;
                beta_i = rho_i * (y_hist(:,i)' * r_dir);
                r_dir = r_dir + s_hist(:,i) * (alpha_corr(i) - beta_i);
             end
        end
        pk = -r_dir;

        % Line Search
        alpha_step = 1.0; c1_armijo = 1e-4; rho_backtrack = 0.5; max_ls_iters = 30; % Increased max_ls_iters
        x_curr_ls = x; 
        f_new = f_old; % Initialize f_new to f_old in case line search fails early
        ls_success = false;

        for ls_iter = 1:max_ls_iters
            x_new_ls_candidate = x_curr_ls + alpha_step * pk;
            x_new_ls_projected = max(lb, min(x_new_ls_candidate, ub)); % Project

            try
                f_new_ls_val = obj_fun(x_new_ls_projected);
                output.funcCount = output.funcCount + 1;
            catch ME_ls_obj
                fprintf('  L-BFGS (box) Line Search iter %d: Error in obj_fun: %s. Reducing step.\n', ls_iter, ME_ls_obj.message);
                alpha_step = rho_backtrack * alpha_step; 
                if alpha_step < 1e-15, break; end 
                continue; 
            end
            
            % Armijo condition: Use projected point for function value, but check against original direction
            % For projected steps, a common check is simply f_new < f_old, or a modified Armijo.
            % Using a simple decrease condition for robustness with projection.
            if f_new_ls_val < f_old - c1_armijo * alpha_step * abs(grad_old' * pk) % Ensure sufficient decrease
                x = x_new_ls_projected; 
                f_new = f_new_ls_val; 
                ls_success = true;
                break;
            end

            alpha_step = rho_backtrack * alpha_step;
            if alpha_step < 1e-15, break; end % Step too small
        end
        
        if ~ls_success
            fprintf('  L-BFGS (box): Line search failed after %d attempts. Trying small projected GD step.\n', max_ls_iters);
            alpha_gd = min(1e-4, H0_diag_val * 0.01); % Very small GD step
            x_gd_candidate = x_curr_ls - alpha_gd * grad_old;
            x_gd_projected = max(lb, min(x_gd_candidate, ub));
            try
                f_gd = obj_fun(x_gd_projected);
                output.funcCount = output.funcCount + 1;
                if f_gd < f_old
                    x = x_gd_projected; f_new = f_gd;
                    fprintf('  L-BFGS (box): Fallback GD step improved objective to %e.\n', f_new);
                    ls_success = true; % Mark as success for update purposes
                else
                    output.message = 'L-BFGS: Line search and GD fallback failed to improve objective.';
                    exitflag = -3; % Line search failed completely
                    x_opt = x_curr_ls; f_opt = f_old; % Return previous best
                    return; % Critical failure, exit function
                end
            catch ME_ls_gd
                 output.message = ['L-BFGS: Line search failed, GD fallback obj_fun error: ', ME_ls_gd.message];
                 exitflag = -4; % Error in fallback
                 x_opt = x_curr_ls; f_opt = f_old;
                 return; % Critical failure, exit function
            end
        end

        % Gradient at new point x
        try
            grad_new = grad_fun_numerical(x);
        catch ME_grad
            output.message = ['L-BFGS: Error in grad_fun_numerical at new x: ', ME_grad.message];
            exitflag = -12; % Gradient computation error
            x_opt = x; f_opt = f_new; % Return current best before error
            return; % Critical failure, exit function
        end

        % Update L-BFGS history
        s_k = x - x_curr_ls;
        y_k = grad_new - grad_old;

        if (s_k' * y_k) > 1e-9 * norm(s_k) * norm(y_k) % More robust curvature condition
            if size(s_hist, 2) >= m_history
                s_hist = s_hist(:, 2:end); y_hist = y_hist(:, 2:end);
            end
            s_hist = [s_hist, s_k]; y_hist = [y_hist, y_k];
        else
             fprintf('  L-BFGS (box) iter %d: Skipping L-BFGS update, y_k^T s_k = %e (not positive enough or s_k/y_k small).\n', iter_count, s_k'*y_k);
        end
        
        % Print progress
        if mod(k, 5) == 0 || k == max_iter -1 || ls_success == false
             fprintf('  L-BFGS (box) iter: %3d, Obj: %e, GradNorm: %e, StepNorm: %e, Alpha: %e\n', ...
                     iter_count, f_new, norm(grad_new), norm(s_k), alpha_step);
        end

        % Check convergence criteria
        if norm(grad_new) < grad_tol 
            output.message = 'L-BFGS: Gradient norm below tolerance.';
            exitflag = 1; break; % Converged
        end
        if norm(s_k) < step_tol * (1 + norm(x_curr_ls)) && ls_success % Only if line search made a step
            output.message = 'L-BFGS: Step size below tolerance.';
            exitflag = 2; break; % Converged (small step)
        end
        if abs(f_new - f_old) < func_tol * (1 + abs(f_old)) && ls_success % Only if line search made a step
            output.message = 'L-BFGS: Function value change below tolerance.';
            exitflag = 3; break; % Converged (small func change)
        end
        
        f_old = f_new;
        grad_old = grad_new;
    end % End L-BFGS loop

    % Assign final outputs if loop completed or broke with a specific exitflag
    if exitflag == -10 % Means loop finished by max_iter and no other exitflag was set
        exitflag = 0; 
        output.message = 'L-BFGS: Maximum iterations reached.';
    end

    x_opt = x; 
    f_opt = f_old; % f_old holds the objective value at the final x
end