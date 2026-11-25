import numpy as np
from scipy.optimize import minimize
from lifelines import KaplanMeierFitter
from dataclasses import dataclass
from scipy.special import betaln
from collections.abc import Callable


@dataclass
class GammaParams:
    k: float
    theta: float

@dataclass
class WeibullParams:
    alpha: float
    lambda_: float

@dataclass
class Baseline:
    """Two parameter baseline cumulative hazard. everything on log scale for numerical stability"""
    name: str
    params_class: type
    log_cum_hazard: Callable
    param_estimation: Callable

def _weibull_log_cum_hazard(t,alpha,log_lambda):
    # H0 = lambda * t**alpha
    return log_lambda + np.log(t)*alpha

def _estimate_weibull_shape_scale_ols(Y, ipcw_weights, k, min_shape=1e-3):
    """
    Estimate initial Weibull (log_a , log_lambda) via OLS on
    log(H0_hat(t)) vs log(t).
    """


    kmf = KaplanMeierFitter()
    kmf.fit(durations=Y, event_observed=ipcw_weights>0)

    # survival_function_ is indexed by timeline (time t) with column 'KM_estimate'
    t = kmf.survival_function_.index.to_numpy(dtype=float)
    s = kmf.survival_function_["KM_estimate"].to_numpy(dtype=float)
    mask_middle = (t > 0) & (s > 0) & (s < 1)
    t, s = t[mask_middle], s[mask_middle]

    # H0_hat(t) = k * (S_hat(t)^(-1/k) - 1) = k * expm1(-log(S_hat(t)) / k)
    h0_hat = k * np.expm1(-np.log(s) / k)

    slope, intercept = np.polyfit(np.log(t), np.log(h0_hat), deg=1)
    log_a = np.log(max(slope, min_shape))
    log_lambda = intercept

    return log_a, log_lambda

def _log_survival(log_k, log_theta, log_H):
    """log of (1+theta * H0(t))**(-k)"""
    return -np.exp(log_k)*np.logaddexp(0.0, log_H + log_theta)

LOG_BOUND = 20

BASELINES ={
    "weibull": Baseline("Weibull", WeibullParams, _weibull_log_cum_hazard, _estimate_weibull_shape_scale_ols)
}



def estimation_gamma_baseline(Y, ipcw_weights, log_k0, log_a0, log_lambda0, baseline="weibull", J=20):
    """
    Estimates the parameters of a gamma distribution given an unknown baseline cumulative hazard with two parameters.
    - Y are the actual event times with no preprocessing.
    - ipcw_weights are the inverse probability of censoring weights for each observation. Should be same size as W. For Harrel's C-index set weights 1 for uncensored and 0 for censored.
    - log_k0: initial estimate of the shape parameter of the gamma distribution.
    - log_lambda0: initial estimate of the scale paramater of the baseline.
    - log_a0: initial estimate of the shape parameter of the baseline distribution.
    - J is the number of quantiles used in the estimation. Requires J >= 2. 
    """

    baseline = BASELINES[baseline]

    Y = np.asarray(Y, dtype=float)
    Y_events = Y[ipcw_weights > 0]
    N = len(Y)#np.sum(ipcw_weights)

    # percentiles used for estimation
    base_percentiles = np.linspace(0.01, 0.95, J)
    y_eval = np.quantile(Y_events, base_percentiles)

    #empirical moments
    m_hat = np.array([np.sum((Y <= y) * ipcw_weights) / N  for y in y_eval])
    m_hat = np.clip(1.0 - m_hat, 1e-8, 1.0)

    #theoretical moments
    def calc_moment(log_k, log_a, log_lambda):
        """Assuming E[v]=1"""
        log_H = baseline.log_cum_hazard(y_eval, np.exp(log_a), log_lambda)
        return np.exp(_log_survival(log_k, -log_k, log_H))

    # optimization objective
    def objective(x, A):
        log_k, log_a, log_lambda = x
        m = calc_moment(log_k, log_a, log_lambda)
        diff = m_hat - m
        return diff.T @ A @ diff


    # wighted matrix to normalize importance of each moment
    A_weights = 1.0 / (m_hat**2 + 1e-8)
    A = np.diag(A_weights)
    
        # initial guess and bounds for optimization
    x0 = np.array([log_k0, log_a0, log_lambda0])
    bounds = [(-LOG_BOUND, LOG_BOUND)]*len(x0)

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

    k_est, a_est, lambda_est = np.exp(result.x)
    theta_est = 1/k_est

    return GammaParams(k=k_est, theta=theta_est), baseline.params_class(alpha=a_est, lambda_=lambda_est)

def random_initialization_gamma_baseline(Y, ipcw_weights, baseline="weibull", subset_frac=0.7):

    baseline = BASELINES[baseline]
    Y = np.asarray(Y)

    N = len(Y)
    idx = np.random.choice(N, size=int(subset_frac * N), replace=True, p=ipcw_weights/np.sum(ipcw_weights))
    Y_sub = Y[idx]
    ipcw_weights_sub = ipcw_weights[idx]
    
    # Randomize the shape parameter
    k_init = np.random.uniform(0.1, 2.0)
    log_k = np.log(k_init)
    log_a, log_lambda = baseline.param_estimation(Y_sub, ipcw_weights_sub, k_init)
    

    return log_k, log_a, log_lambda


def ecindex_baseline(T, C, num_samples=100, baseline="weibull",
                         fraction_initialization=0.7, J=20):
    """
    Draws num_samples estimates of the expected c-index, each from its own random initialization.
    - T are the observed times, C the event indicator.
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


    for _ in range(num_samples):
        # Randomly initialize parameters
        log_k0, log_a0, log_lambda0 = random_initialization_gamma_baseline(T, ipcw_weights,baseline=baseline, subset_frac=fraction_initialization)
        # Estimate parameters
        gamma_param, weibull_param = estimation_gamma_baseline(T, ipcw_weights, log_k0, log_a0, log_lambda0, baseline=baseline, J=J)
        gamma_params.append(gamma_param)
        # Calculate expected C-index
        ecindex_samples.append(expected_cindex_gamma(gamma_param))

    return ecindex_samples, gamma_params




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
    base_percentiles = np.linspace(0.01, 0.95, J)
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
    base_percentiles = np.linspace(0.01, 0.95, J)
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


def random_initialization_gamma_weibull_groups(Y, ipcw_weights, Sub, subset_frac=0.7):
    """
    Random initial parameters for the subpopulation estimator.
    For each group, a weighted random subset of that group is drawn, and the group's
    survival curve is anchored so that it passes through the subset's median:
        (1 + theta_g * lambda * y_med_g^a)^(-k_g) = 0.5
    The Weibull shape a is shared. The reference group has theta_ref = 1/k_ref,
    which fixes lambda; the other groups' theta_g follow from their own medians.

    Returns k0s (dict), lambda0, a0, theta0s (dict, not for first group),
    in the format expected by estimation_gamma_weibull.
    """
    Y = np.asarray(Y, dtype=float)
    w = np.asarray(ipcw_weights, dtype=float)
    Sub = np.asarray(Sub)

    groups = list(np.unique(Sub))

    ref_group = groups[0]

    # shared Weibull shape
    a_init = np.random.uniform(0.1, 2.0)

    k0s, combined = {}, {}
    for g in groups:
        mask = Sub == g
        Y_g, w_g = Y[mask], w[mask]
        n_g = len(Y_g)
        idx = np.random.choice(n_g, size=max(1, int(subset_frac * n_g)),
                               replace=True, p=w_g / np.sum(w_g))
        y_med = np.median(Y_g[idx])

        k_g = np.random.uniform(0.1, 2.0)
        # combined scale theta_g * lambda that puts the median of S_g at y_med
        c_g = ((2.0 ** (1.0 / k_g)) - 1.0) / (y_med ** a_init)
        k0s[g] = k_g
        combined[g] = max(c_g, 1e-6)

    # reference group: theta_ref = 1/k_ref  =>  lambda = c_ref * k_ref
    lambda0 = combined[ref_group] * k0s[ref_group]
    theta0s = {g: max(combined[g] / lambda0, 1e-6) for g in groups if g != ref_group}

    return k0s, lambda0, a_init, theta0s

def estimation_gamma_weibull_groups(Y, ipcw_weights, Sub, k0s, lambda0, a0,
                             theta0s=None, J=20):
    """
    Gamma frailty per subpopulation with a shared Weibull baseline H0(t) = lambda * t**a.
    Identifiability: only the reference group has mean frailty 1 (theta_ref = 1/k_ref).
    The other groups have free (k_g, theta_g); their mean frailty k_g * theta_g is their
    risk relative to the reference group.
    Model: S_g(t) = (1 + theta_g * lambda * t**a) ** (-k_g)

    - k0s: dict {group: initial k}
    - lambda0: initial Weibull scale
    - a0: initial Weibull shape
    - theta0s: optional dict {group: initial theta} for the non-reference groups
      (default 1/k0, i.e. start from mean frailty 1)
    """
    Y = np.asarray(Y, dtype=float)
    w = np.asarray(ipcw_weights, dtype=float)
    Sub = np.asarray(Sub)

    groups = list(np.unique(Sub))

    ref_group = groups[0]
    groups.remove(ref_group)
    groups = [ref_group] + groups# reference is always at position 0
    n_sub = len(groups)

    base_percentiles = np.linspace(0.01, 0.95, J)

    # empirical moments, stacked in the order of `groups`
    y_evals, m_hat = [], []
    for g in groups:
        mask = Sub == g
        Y_g, w_g = Y[mask], w[mask]
        y_g = np.quantile(Y_g[w_g > 0], base_percentiles)
        F_g = np.array([np.sum((Y_g <= y) * w_g) for y in y_g]) / mask.sum()
        y_evals.append(y_g)
        m_hat.append(np.clip(1.0 - F_g, 1e-8, 1.0))
    m_hat = np.concatenate(m_hat)

    # x = [k_ref, k_2, ..., k_G, lambda, a, theta_2, ..., theta_G]
    def unpack(x):
        ks = x[:n_sub]
        lam, a = x[n_sub], x[n_sub + 1]
        thetas = np.concatenate([[1.0 / ks[0]], x[n_sub + 2:]])
        return ks, thetas, lam, a

    def calc_moment(ks, thetas, lam, a):
        return np.concatenate([
            (1.0 + thetas[j] * lam * y_evals[j] ** a) ** (-ks[j])
            for j in range(n_sub)
        ])

    def objective(x, A):
        ks, thetas, lam, a = unpack(x)
        diff = m_hat - calc_moment(ks, thetas, lam, a)
        return diff @ A @ diff

    A = np.diag(1.0 / (m_hat ** 2 + 1e-8))

    if theta0s is None:
        theta0s = {g: 1.0 / k0s[g] for g in groups[1:]}
    x0 = np.array([k0s[g] for g in groups] + [lambda0, a0]
                  + [theta0s[g] for g in groups[1:]], dtype=float)
    bounds = [(1e-5, None)] * len(x0)

    result = minimize(objective, x0, args=(A,), bounds=bounds, method="SLSQP",
                      options={"maxiter": 1000, "ftol": 1e-8})
    if not result.success:
        print(f"Warning: Optimizer did not fully converge. Message: {result.message}")

    ks, thetas, lam, a = unpack(result.x)
    gammas = {g: GammaParams(k=ks[j], theta=thetas[j]) for j, g in enumerate(groups)}
    return gammas, WeibullParams(alpha=a, lambda_=lam)


def ecindex_gamma_mc(gammas, sub_weights, num_samples=1000, group = None):
    aux = []
    while len(aux) < num_samples:
      if group is None:
        t1 = np.random.choice(list(sub_weights.keys()), p=list(sub_weights.values()))
        t2 = np.random.choice(list(sub_weights.keys()), p=list(sub_weights.values()))
      else:
        t1 = group
        t2 = np.random.choice(list(sub_weights.keys()), p=list(sub_weights.values()))
      
      v1 = np.random.gamma(gammas[t1].k, gammas[t1].theta)
      v2 = np.random.gamma(gammas[t2].k, gammas[t2].theta)
      if v1 + v2 <1e-7:
          continue
      elif v1 > v2:
        aux.append(v1/(v1+v2))
      else:
        aux.append(v2/(v1+v2))
    return np.mean(aux)

def ecindex_distribution_groups(T, C, Sub, num_samples=100,
                         fraction_initialization=0.7, J=20, mc_samples=1000, group = None):
    """
    Draws num_samples estimates of the expected c-index, each from its own random initialization.
    - T are the observed times, C the event indicator.
    - cumulative_baseline_hazard, if given, transforms T so the baseline is known and only the gamma
      has to be estimated. If None the baseline is taken to be an unknown weibull.
    - num_samples is the number of random initializations, ie the size of the returned sample.
    - fraction_initialization is the subset fraction used by the initializers.
    Returns the sample of expected c-indices and the matching gamma parameters.
    - J is the number of quantiles used in the estimation. Requires J >= 2.
    - mc_samples is the number of samples used to estimate the expected c-index using Monte Carlo.
    - group: use to calculate expected cindex of an specific group.
    TODO: Implement with known cumulative baseline hazard.
    """
    gamma_params = []
    ecindex_samples = []
    G_fitter = KaplanMeierFitter()
    G_fitter.fit(durations=T, event_observed=(1 - C))
    G_times = G_fitter.survival_function_at_times(T).to_numpy()
    ipcw_weights = C/G_times
    
    groups = list(np.unique(Sub))
    sub_weights = {i: np.mean(Sub==i) for i in groups}


    # Administrative censoring to 0.
    mask = np.isnan(ipcw_weights)
    ipcw_weights[mask] = 0


    for _ in range(num_samples):
        # Randomly initialize parameters
        k0s, lam0, a0, th0s = random_initialization_gamma_weibull_groups(T, ipcw_weights, Sub, subset_frac=fraction_initialization)
        # Estimate parameters
        gammas, weibull = estimation_gamma_weibull_groups(T, ipcw_weights, Sub, k0s, lam0, a0, theta0s=th0s, J=J)            
        
        gamma_params.append(gammas)
        # Calculate expected C-index
        ecindex_samples.append(ecindex_gamma_mc(gammas, sub_weights, num_samples=mc_samples, group = group))

    return ecindex_samples, gamma_params