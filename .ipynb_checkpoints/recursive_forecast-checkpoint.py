"""
    Recursive forecast
    Parameters:
        n: number of preceding points to look back
    Formula;
        y_n = f(y_0,...,y_n)
            = f(y)          y is shorthanded notation for column vector {y_0, y_1, ..., y_n}.T
            ≈ f(0) + y*(grad_f(0)) + 0.5 * y.T * Hessian_f(0) * y        (2nd-order Taylor approximation)
"""