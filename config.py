MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

MC_RUNS = 15
MAX_NEW_TOKENS = 128

TEMPERATURE = 0.7
TOP_P = 0.9

ALPHA = 0.4   # mean entropy weight
BETA = 0.3    # entropy spike weight
GAMMA = 0.3   # semantic variance weight

LAMBDA = 1.2  # confidence decay
