import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 初始化一个足够长的 位置编码矩阵 self.P
        self.P = torch.zeros((1, max_len, num_hiddens))

        A = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        B = torch.arange(start=0, end=num_hiddens, step=2, dtype=torch.float32)
        C = B / num_hiddens

        X = A / torch.pow(10000, C)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        # 只取与输入序列长度相匹配的位置编码
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
