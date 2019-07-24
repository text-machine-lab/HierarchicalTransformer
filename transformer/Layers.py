''' Define the Layers '''
import torch.nn as nn
import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward

__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class UNetEncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, type='same'):
        super(UNetEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)

        self.type = type
        if type == 'down':
            # half size of output
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        elif type == 'same':
            # keep size of output the same
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        elif type == 'up':
            # double size of output
            self.conv = nn.ConvTranspose1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        else:
            raise RuntimeError('Did not specify appropriate convolution type')

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None, ctx_input=None):
        #TODO in output of unet, allow attention over both input and input layer of same size

        # we assume enc_input is of shape (batch_size, n_steps, emb_size)
        # we run convolution/deconvolution over input to up or down sample (or keep same size)
        assert enc_input is not None or ctx_input is not None
        # this combines signals during unet upsampling phase
        conv_input = 0.0
        if ctx_input is not None:
            conv_input += ctx_input
        if enc_input is not None:
            conv_input += enc_input

        conv_output = conv_input.transpose(1, 2)  # (batch_size, emb_size, n_steps)

        if self.type != 'up':
            conv_output = self.conv(conv_output).transpose(1, 2)
        else:
            # we are doing a transpose
            output_size = None if non_pad_mask is None else \
                (non_pad_mask.shape[0], conv_input.shape[2], non_pad_mask.shape[1])

            conv_output = self.conv(conv_output, output_size=output_size).transpose(1, 2)

        # if same, we use skip connections from input to output to allow for more efficient gradient propagation
        if self.type == 'same':
            conv_output = self.norm(enc_input + conv_output)
        else:
            conv_output = self.norm(conv_output)

        # here the output of the convolution performs attention over the input
        enc_output, enc_slf_attn = self.slf_attn(
            conv_output, conv_input, conv_input, mask=slf_attn_mask)

        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn
