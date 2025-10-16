import torch
from d2l import torch as d2l
from model.encoder import TransformerEncoder
from model.decoder import TransformerDecoder


num_hiddens, num_blks, dropout = 256, 2, 0.2
ffn_num_hiddens, num_heads = 64, 4

data = d2l.MTFraEng(batch_size=128)

encoder = TransformerEncoder(len(data.src_vocab), num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout)
decoder = TransformerDecoder(len(data.tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout)
model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'], lr=0.001)
trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=0)
trainer.fit(model, data)

engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']

preds, _ = model.predict_step(batch=data.build(engs, fras), device=d2l.try_gpu(), num_steps=data.num_steps)
for en, fr, p in zip(engs, fras, preds):
    translation = []
    # 将预测的token ID转换回文本token
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)
    print(f'{en} => {translation}, bleu, 'f'{d2l.bleu(" ".join(translation), fr, k=2):.3f}')

