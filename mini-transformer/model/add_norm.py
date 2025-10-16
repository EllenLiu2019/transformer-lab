import torch.nn as nn

class AddNorm(nn.Module):  #@save
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    # X 是原始输入（残差连接的"捷径"部分）OR "之前的小net"
    # Y 是经过子层（如注意力或前馈网络）处理后的输出
    def forward(self, X, Y):
        residual_conn = self.dropout(Y) + X
        return self.ln(residual_conn)