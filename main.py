from model_loader import load_model
from inference.mc_sampler import monte_carlo_generate
from uncertainty.entropy import mean_entropy, entropy_spike
from uncertainty.semantic_variance import semantic_variance
from uncertainty.fusion import hallucination_score
from calibration.confidence import confidence_score, risk_label

def run_pipeline(question):
    tokenizer, model = load_model()

    outputs, entropies = monte_carlo_generate(model, tokenizer, question)

    mean_ent = mean_entropy(entropies)
    spike_ent = entropy_spike(entropies)
    sem_var = semantic_variance(outputs)

    h_score = hallucination_score(mean_ent, spike_ent, sem_var)
    conf = confidence_score(h_score)
    risk = risk_label(conf)

    print("\n==============================")
    print("ANSWER:")
    print(outputs[0])
    print("\n--- Reliability ---")
    print(f"Mean Entropy      : {mean_ent:.4f}")
    print(f"Entropy Spike     : {spike_ent:.4f}")
    print(f"Semantic Variance : {sem_var:.4f}")
    print(f"\nConfidence Score (C): {conf:.3f}")
    print(f"Hallucination Risk : {risk}")
    print("==============================")

if __name__ == "__main__":
    question = "According to standard clinical guidelines, what are the recommended first-line treatments for hypertension in adults?"

    run_pipeline(question)
