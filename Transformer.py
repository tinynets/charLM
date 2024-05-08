import torch.nn as nn
from EncoderBlock import Encoder
from DecoderBlock import Decoder


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_encoder_layers, n_decoder_layers):
        super(Transformer, self).__init__()
        self.encoders = nn.ModuleList([Encoder(vocab_size, d_model) for _ in range(n_encoder_layers)])
        self.decoders = nn.ModuleList([Decoder(vocab_size, d_model) for _ in range(n_decoder_layers)])
        self.final_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, trg):
        enc_output = src
        for encoder in self.encoders:
            enc_output = encoder(enc_output)

        dec_output = trg
        for decoder in self.decoders:
            dec_output = decoder(dec_output, enc_output)

        logits = self.final_layer(dec_output)
        output = nn.functional.log_softmax(logits, dim=-1)

        return output