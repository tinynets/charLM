import torch.nn as nn
from EncoderBlock import Encoder
from DecoderBlock import Decoder
from PositionalEncoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_encoder_layers, n_decoder_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=50)
        self.encoders = nn.ModuleList([Encoder(vocab_size, d_model) for _ in range(n_encoder_layers)])
        self.decoders = nn.ModuleList([Decoder(vocab_size, d_model) for _ in range(n_decoder_layers)])
        self.final_layer = nn.Linear(d_model, vocab_size)

    def forward(self, source, target):

        source = self.embedding(source)
        source = self.pos_enc(source)

        enc_output = source
        for encoder in self.encoders:
            enc_output = encoder(enc_output)

        # we are done with the encoder part, on to decoder

        print(f'encoder part done, enc_output shape: {enc_output.shape}')
        print(target.shape)

        dec_output = target
        for decoder in self.decoders:
            dec_output = decoder(dec_output, enc_output)

        print('final dec_output')
        print(dec_output.shape)
        logits = self.final_layer(dec_output)
        output = nn.functional.log_softmax(logits, dim=-1)

        return output