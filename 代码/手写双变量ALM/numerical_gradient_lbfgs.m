% numerical_gradient_lbfgs.m
function grad = numerical_gradient_lbfgs(fun_handle, x, h_diff)
% Calculates numerical gradient using central differences.
% fun_handle: Handle to the function whose gradient is needed (e.g., augmented Lagrangian).
% x: Point at which to evaluate the gradient.
% h_diff: Step size for finite difference.

    n = length(x);
    grad = zeros(n, 1);
    
    if nargin < 3 || isempty(h_diff)
        h_diff = 1e-7; % Default step size if not provided
    end

    for i = 1:n
        x_plus_h = x;
        x_minus_h = x;
        
        x_plus_h(i) = x_plus_h(i) + h_diff;
        x_minus_h(i) = x_minus_h(i) - h_diff;
        
        try
            f_plus_h = fun_handle(x_plus_h);
            f_minus_h = fun_handle(x_minus_h);
        catch ME
            fprintf('Error in numerical_gradient_lbfgs when evaluating fun_handle at x_plus_h or x_minus_h for index %d.\n', i);
            fprintf('x(i) = %e, h_diff = %e\n', x(i), h_diff);
            fprintf('Error message: %s\n', ME.message);
            rethrow(ME); % Re-throw the error to stop execution
        end
        
        grad(i) = (f_plus_h - f_minus_h) / (2 * h_diff);
    end
end