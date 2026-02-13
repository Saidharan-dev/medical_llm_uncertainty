import numpy as np

def mean_entropy(entropy_lists):
    flat = [np.mean(e) for e in entropy_lists]
    return float(np.mean(flat))

def entropy_spike(entropy_lists, percentile=95):
    flat = [val for sub in entropy_lists for val in sub]
    threshold = np.percentile(flat, percentile)
    spikes = [e for e in flat if e >= threshold]
    return float(np.mean(spikes)) if spikes else 0.0
