import torch
import torch.nn.functional as F
from tokenizer import encode, decode
from model import MiniLLM
from config import config

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

model = MiniLLM(
    vocab_size=config["vocab_size"],
    d_model=config["d_model"],
    n_layers=config["n_layers"],
    n_heads=config["n_heads"],
    n_kv_heads=config["n_kv_heads"],
    ffn_hidden_dim=config["ffn_hidden_dim"],
    max_seq_len=config["max_seq_len"]
).to(device)

model.load_state_dict(
    torch.load("checkpoints/best_model.pth", map_location=device)
)

model.eval()

@torch.no_grad()
def generate(model, prompt, max_new_tokens=500, temperature=0.8):
    model.eval()
    tokens = encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        context = tokens[:, -config["max_seq_len"]:]
        logits, _ = model(context)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return decode(tokens[0].tolist())

if __name__ == "__main__":

    prompt = "ROMEO:"
    print("=" * 60)
    print(f"PROMPT: {prompt!r}")
    print("=" * 60)

    for temp in [0.5, 0.8, 1.0, 1.2]:
        print("\n" + "_" * 60)
        print(f"Temperature = {temp}")
        print("_" * 60)
        output = generate(model, prompt, max_new_tokens=300, temperature=temp)
        print(output)

    prompts = [""]

    for p in prompts:
        print("\n" + "=" * 60)
        print(f"PROMPT: {p!r}")
        print("=" * 60)
        output = generate(model, p, max_new_tokens=200, temperature=0.8)
        print(output)