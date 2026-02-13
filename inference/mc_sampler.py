import torch
import torch.nn.functional as F
import numpy as np
from config import MC_RUNS, TEMPERATURE, TOP_P, MAX_NEW_TOKENS

def monte_carlo_generate(model, tokenizer, prompt):
    all_outputs = []
    all_entropies = []

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    for _ in range(MC_RUNS):
        with torch.no_grad():
            result = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                output_scores=True,
                return_dict_in_generate=True
            )

        text = tokenizer.decode(result.sequences[0], skip_special_tokens=True)
        all_outputs.append(text)

        token_entropies = []
        for scores in result.scores:
            probs = F.softmax(scores, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
            token_entropies.append(entropy.mean().item())

        all_entropies.append(token_entropies)

    return all_outputs, all_entropies
