import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .add_norm import AddNorm
from .postionwise_FFN import PositionWiseFFN

class TransformerDecoderBlock(nn.Module):
    # The i-th block in the Transformer decoder 第i个块
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, X, state):
        # 从 state 中提取编码器输出和有效长度
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed at the same time, so state[2][self.i] is None as initialized.
        # When decoding any output sequence token by token during prediction
        # state[2][self.i] contains representations of the decoded output at the i-th block up to the current time step

        # 训练阶段，输出序列的所有词元都在同一时间处理， 因此 state[2][self.i] 初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None: # 第一个时间步
            key_values = X
        else:
            # 将 上一个时间步的 X 与 当前时间步的 X 在 1 维拼接起来 ==》 截止到当前时间步的所有输入 X的拼接 ==》 只 `关注` 解码器中 ‘截止到该查询位置为止’ 所有已经生成的 token
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # Shape of dec_valid_lens: (batch_size, num_steps), where every row is [1, 2, ..., num_steps]
            # training 时 只 `关注` 解码器中 ‘截止到该查询位置为止’ 所有已经生成的 token
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # Masked self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)

        # Encoder-decoder attention. Shape of enc_outputs: (batch_size, num_steps, num_hiddens)
        # enc_outputs： 编码器的输出
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)

        # Positionwise feed-forward network
        return self.addnorm3(Z, self.ffn(Z)), state