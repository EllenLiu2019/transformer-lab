import torch.nn as nn

class PositionWiseFFN(nn.Module):  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        # 第一个全连接层 (dense1)
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 第二个全连接层 (dense2)
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))