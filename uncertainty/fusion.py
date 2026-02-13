from config import ALPHA, BETA, GAMMA

def hallucination_score(mean_ent, spike_ent, semantic_var):
    return (
        ALPHA * mean_ent +
        BETA * spike_ent +
        GAMMA * semantic_var
    )
