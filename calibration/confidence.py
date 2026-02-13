import numpy as np
from config import LAMBDA

def confidence_score(h_score):
    return float(np.exp(-LAMBDA * h_score))

def risk_label(conf):
    if conf >= 0.85:
        return "Low"
    elif conf >= 0.60:
        return "Medium"
    else:
        return "High"
