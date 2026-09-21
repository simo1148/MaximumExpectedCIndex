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
