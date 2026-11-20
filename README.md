# Maximum Expected Cindex

## Usage

```python
import ecindex

durations = np.array([5, 6, 6, 2, 4, 4])
observed = np.array([1, 0, 1, 1, 1, 0])
C = observed == 1 # censoring flags: True if event observed, False if censored

# returns  expected c-index + sorted non-censored events with corresponding hazards and weights.
ec, events, hazards, weights = ecindex.ecindex(durations, C, t_star = np.max(durations)+1)

```

# Distribution of Expected C-Index

This version estimates the best Gamma distribution that fits the data. If a cumulative hazard is known then it can be passed to the solver. If it isn't then it estimates the baseline using a Weibull distribution. 
## Usage
```
ec_dist, gamma_params = dist.ecindex_distribution(durations, C, cumulative_baseline_hazard=None, num_samples=100, fraction_initialization=0.7, J=20)
```

