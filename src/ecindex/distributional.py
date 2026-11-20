import numpy as np
from scipy.optimize import minimize
from lifelines import KaplanMeierFitter
from dataclasses import dataclass
from scipy.special import betaln




@dataclass
class GammaParams:
    k: float
    theta: float

@dataclass
class WeibullParams:
    alpha: float
    lambda_: float



def estimation_gamma_cens(W, ipcw_weights, k0, theta0, J=20):
    """
    Estimates the parameters of a gamma distribution given a known baseline distribution. 
    - Assumes that the W are already transformed via the cumulative baseline hazard function and thus W is exponentially distributed with mean drawn from a gamma distribution. 
    - ipcw_weights are the inverse probability of censoring weights for each observation. Should be same size as W. For Harrel's C-index set weights 1 for uncensored and 0 for censored. 
    - k0 and theta0 are the initial estimates for the shape and scale parameters of the gamma distribution. 
    - J is the number of quantiles used in the estimation. Requires J >= 2. 
    """

    if len(W) != len(ipcw_weights):
        raise ValueError("W and ipcw_weights must be the same length.")
    if J < 2:
        raise ValueError("J must be at least 2.")

    W = np.asarray(W, dtype=float)
    W_events = W[ipcw_weights > 0]  # Only consider events for estimation
    N = len(W)#np.sum(ipcw_weights)

    # percentiles used for estimation
    base_percentiles = np.linspace(np.sqrt(0.01), np.sqrt(0.95), J)
    w_eval = np.quantile(W_events, base_percentiles)

    #empirical moments
    m_hat = np.array([np.sum((W <= w_val) * ipcw_weights) / N  for w_val in w_eval])
    m_hat = np.clip(1.0 - m_hat, 1e-8, 1.0)

    # theoretical moments
    def calc_moment(k, theta):
        return (1.0 + theta * w_eval) ** (-k)

    # optimization objective
    def objective(x, A):
        k, theta = x
        m = calc_moment(k, theta)
        diff = m_hat - m
        return diff.T @ A @ diff

    # wighted matrix to normalize importance of each moment
    A_weights = 1.0 / (m_hat**2 + 1e-8)
    A = np.diag(A_weights)

    # initial guess and bounds for optimization
    x0 = np.array([k0,theta0])
    bounds = [(1e-5, None), (1e-5, None)]

    result = minimize(
        objective,
        x0,
        bounds=bounds,
        args=(A,),
        method='SLSQP',
        options={'maxiter': 1000, 'ftol': 1e-8}
    )

    if not result.success:
        print(f"Warning: Optimizer did not fully converge. Message: {result.message}")

    k_est, theta_est = result.x
    return GammaParams(k=k_est, theta=theta_est)


    
def random_initialization_gamma(W, fraction=0.7):
    """
    Randomly samples a fraction of the data to compute initial estimates for the gamma distribution parameters, based on the moments of a lomax distribution, ie mixture of exponentials.
    Harris, C. M. (1968). The Pareto distribution as a queue service discipline. Operations Research, 16(2), 307-313.
    """

    sample = np.random.choice(W, size = int(len(W) * fraction) )
    N = len(sample)
    expectation = np.sum(sample) / N
    second_moment = np.sum((sample ** 2)) / N

    R = second_moment / (2.0 * expectation**2 + 1e-8)

    k_init = (2.0 * R - 1.0) / (R - 1.0)
    theta_init = 1.0 / (expectation * (k_init - 1.0))
    return GammaParams(k=k_init, theta=theta_init)


def random_initialization_gamma_weibull(Y, ipcw_weights, subset_frac=0.7):
    """
    Generates initial parameters by taking a random subset of the data
    and anchoring the scale parameter to the subset's empirical median.
    """
    Y = np.asarray(Y)

    N = len(Y)
    idx = np.random.choice(N, size=int(subset_frac * N), replace=True, p=ipcw_weights/np.sum(ipcw_weights))
    Y_sub = Y[idx]
    
    y_med = np.median(Y_sub)
    
    # Randomize the shape parameters
    k_init = np.random.uniform(0.1, 2.0)
    a_init = np.random.uniform(0.1, 2.0)

    # CALCULATE gamma based on the subset
    # We force the theoretical survival curve to perfectly hit the subset's median:
    # (1 + gamma * y_med^a)^(-k) = 0.5  =>  gamma = (2^(1/k) - 1) / y_med^a
    theta_lambda = ( (2.0 ** (1.0 / k_init)) - 1.0 ) / (y_med ** a_init)

    # Clip gamma to prevent numerical zeros if y_med is massive
    theta_lambda_init = max(theta_lambda, 1e-6)

    return k_init, theta_lambda_init, a_init


def expected_cindex_gamma(gamma_params: GammaParams):
    """
    Exactly calculates E[ v_i / (v_i + v_j) | v_i > v_j ] 
    for v_i, v_j ~ Gamma(k, theta).
    
    Uses log-space for numerical stability at large values of k.
    """
    k =gamma_params.k
    
    # Calculate the second term in log-space: 
    # ln( 4^-k / (k * B(k,k)) ) = -k*ln(4) - ln(k) - ln(B(k,k))
    log_term = -k * np.log(4.0) - np.log(k) - betaln(k, k)
    
    return 0.5 + np.exp(log_term)

def estimation_gamma_weibull(Y, ipcw_weights, k0, theta_lambda0, a0, J=20):
    """
    Estimates the parameters of a gamma distribution given an unknown baseline cumulative hazard which is assume to be weibull with unknown parameters.
    To make the problem identifiable we assume that the mean of the gamma distribution is 1.0 and we combine the theta parameter of the gamma with the lambda parameter of the weibull. We can then recover the theta by 1/k.
    In this case, the 
    - Y are the actual event times with no preprocessing.
    - ipcw_weights are the inverse probability of censoring weights for each observation. Should be same size as W. For Harrel's C-index set weights 1 for uncensored and 0 for censored.
    - k0: initial estimate of the shape parameter of the gamma distribution.
    - theta_lambda0: initial estimate of the combined scale parameter of the gamma and weibull distribution.
    - a0: initial estimate of the shape parameter of the weibull distribution.
    - J is the number of quantiles used in the estimation. Requires J >= 2. 
    """
    Y = np.asarray(Y, dtype=float)
    Y_events = Y[ipcw_weights > 0]
    N = len(Y)#np.sum(ipcw_weights)

    # percentiles used for estimation
    base_percentiles = np.linspace(np.sqrt(0.01), np.sqrt(0.95), J)
    y_eval = np.quantile(Y_events, base_percentiles)

    #empirical moments
    m_hat = np.array([np.sum((Y <= y) * ipcw_weights) / N  for y in y_eval])
    m_hat = np.clip(1.0 - m_hat, 1e-8, 1.0)

    #theoretical moments
    def calc_moment(k, theta_lambda, a):
        return (1.0 + theta_lambda * (y_eval ** a)) ** (-k)

    # optimization objective
    def objective(x, A):
        k, theta_lambda, a = x
        m = calc_moment(k, theta_lambda, a)
        diff = m_hat - m
        return diff.T @ A @ diff


    # wighted matrix to normalize importance of each moment
    A_weights = 1.0 / (m_hat**2 + 1e-8)
    A = np.diag(A_weights)
    
        # initial guess and bounds for optimization
    x0 = np.array([k0, theta_lambda0, a0])
    bounds = [(1e-5, None), (1e-5, None), (1e-5, None)]

    result = minimize(
        objective,
        x0,
        bounds=bounds,
        args=(A,),
        method='SLSQP',
        options={'maxiter': 1000, 'ftol': 1e-8}
    )

    if not result.success:
        print(f"Warning: Optimizer did not fully converge. Message: {result.message}")

    k_est, theta_lambda_est, a_est = result.x
    theta_est = 1/k_est
    lambda_est = theta_lambda_est/theta_est

    return GammaParams(k=k_est, theta=theta_est), WeibullParams(alpha=a_est, lambda_=lambda_est)
    

def ecindex_distribution(T, C, cumulative_baseline_hazard=None, num_samples=100,
                         fraction_initialization=0.7, J=20):
    """
    Draws num_samples estimates of the expected c-index, each from its own random initialization.
    - T are the observed times, C the event indicator.
    - cumulative_baseline_hazard, if given, transforms T so the baseline is known and only the gamma
      has to be estimated. If None the baseline is taken to be an unknown weibull.
    - num_samples is the number of random initializations, ie the size of the returned sample.
    - fraction_initialization is the subset fraction used by the initializers.
    Returns the sample of expected c-indices and the matching gamma parameters.
    - J is the number of quantiles used in the estimation. Requires J >= 2.
    """
    gamma_params = []
    ecindex_samples = []
    G_fitter = KaplanMeierFitter()
    G_fitter.fit(durations=T, event_observed=(1 - C))
    G_times = G_fitter.survival_function_at_times(T).to_numpy()
    ipcw_weights = C/G_times


    # Administrative censoring to 0.
    mask = np.isnan(ipcw_weights)
    ipcw_weights[mask] = 0


    if cumulative_baseline_hazard is None:
        for _ in range(num_samples):
            # Randomly initialize parameters
            k_init, theta_lambda_init, a_init = random_initialization_gamma_weibull(T, ipcw_weights, subset_frac=fraction_initialization)
            # Estimate parameters
            gamma_param, weibull_param = estimation_gamma_weibull(T, ipcw_weights, k_init, theta_lambda_init, a_init, J=J)
            gamma_params.append(gamma_param)
            # Calculate expected C-index
            ecindex_samples.append(expected_cindex_gamma(gamma_param))
    else:
        for _ in range(num_samples):
            # Randomly initialize parameters
            Y = cumulative_baseline_hazard(T)
            gamma_param = random_initialization_gamma(Y, fraction=fraction_initialization)
            # Estimate parameters
            gamma_param = estimation_gamma_cens(Y, ipcw_weights, gamma_param.k, gamma_param.theta, J = J)
            gamma_params.append(gamma_param)
            # Calculate expected C-index
            ecindex_samples.append(expected_cindex_gamma(gamma_param))
    return ecindex_samples, gamma_params


