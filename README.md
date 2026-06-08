🧠 Mini LLM from Scratch (PyTorch)

A decoder-only Transformer language model built entirely from scratch in PyTorch.

This project implements a modern LLM architecture with components like RoPE, RMSNorm, SwiGLU, and Grouped Query Attention (GQA) and trains on the Tiny Shakespeare dataset.

🚀 Features
🧱 Decoder-only Transformer (GPT-style architecture)
⚡ RMSNorm (instead of LayerNorm)
🔁 Rotary Positional Embeddings (RoPE)
🧠 SwiGLU Feed-Forward Network
🎯 Grouped Query Attention (GQA)
✂️ Gradient clipping for stable training
🍎 Apple Silicon (MPS) support
📉 Train + validation loss tracking
🔤 Character-level tokenizer (BPE-ready upgrade path)
📊 Dataset
Tiny Shakespeare dataset
Character-level tokenization (simple baseline)
Easily extensible to BPE / SentencePiece
🏗️ Model Architecture
Token Embedding → Transformer Blocks → RMSNorm → Linear Head

Each Transformer Block contains:

RMSNorm
Grouped Query Attention (with RoPE)
SwiGLU Feedforward Network
Residual connections
⚙️ Training Details
Batch size: 64
Context length: 256
Optimizer: AdamW
Learning rate: 3e-4
Max steps: 3000
Evaluation interval: every 250 steps
📈 Results

After training on Tiny Shakespeare:

Train loss: ~1.0
Validation loss: ~1.5 – 4.0 (depends on run / dataset size)

Note: Model is character-level, so outputs are not semantically perfect but show learning of structure and style.

💬 Sample Generation
Prompt: ROMEO:
ROMEO:
I do but see thee here and live to light.

Model learns:

Shakespeare-style structure
Character naming patterns
Basic grammar continuity
🧪 How to Run
1. Install dependencies
pip install -r requirements.txt
2. Train model
python train.py
3. Generate text
python generate.py
🔮 Future Improvements
🔥 Replace character tokenizer with BPE / SentencePiece
📚 Train on WikiText-2 / OpenWebText
⚡ Add mixed precision training (AMP)
🧠 Scale model size (more layers + heads)
💾 Add model checkpointing + resume training
🌐 Web UI for text generation
🏁 Goal of this Project

This project was built to understand:

How modern LLMs (GPT-style) actually work
Transformer internals from scratch
Training stability techniques (RoPE, RMSNorm, GQA)