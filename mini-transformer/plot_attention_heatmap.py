from model.attention import MultiHeadAttention
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

def plot_attention(batch_size=1, seq_len=5, d_model=8, num_hiddens=16, num_heads=2):
    # Generate random input tensor
    X = torch.randn(batch_size, seq_len, d_model)

    # Initialize multi-head attention module
    mha = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads, dropout=0.5)

    # Forward pass with gradient disabled for inference
    with torch.no_grad():
      mha(X, X, X, None)

    print(f"Attention weights shape: {mha.attention.attention_weights.shape}")
    print(f"Attention weights:\n{mha.attention.attention_weights}")

    # Prepare attention weights for visualization
    attention_weights = mha.attention.attention_weights.unsqueeze(0)
    d2l.show_heatmaps(attention_weights, xlabel='head', ylabel='Queries', figsize=(10, 5))
    plt.show()


if __name__ == "__main__":
    plot_attention()