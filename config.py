config = {
    "vocab_size": 65,
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 8,
    "n_kv_heads": 2,
    "ffn_hidden_dim": 680,
    "max_seq_len": 256,
}

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MAX_STEPS = 3000
EVAL_INTERVAL = 250
EVAL_STEPS = 20
LOG_INTERVAL = 50
DROPOUT = 0.2