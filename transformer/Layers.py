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
    def __init__(self, d_out, d_inner, n_head, d_k, d_v, dropout=0.1, type_='same', skip_connect=False, d_in=None):
        super(UNetEncoderLayer, self).__init__()

        d_in = d_in if d_in is not None else d_out  # size of input to unet layer

        # TODO input and output sizes of attention are the same now
        self.slf_attn = MultiHeadAttention(
            n_head, d_out, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_out, d_inner, dropout=dropout)

        self.norm = nn.LayerNorm(d_out)

        self.skip_connect = skip_connect

        self.type = type_
        if type_ == 'down':
            # half size of output
            self.conv = nn.Conv1d(d_in, d_out, kernel_size=3, stride=2, padding=1)
        elif type_ == 'same':
            # keep size of output the same
            self.conv = nn.Conv1d(d_in, d_out, kernel_size=3, padding=1)
        elif type_ == 'up':
            # double size of output
            self.conv = nn.ConvTranspose1d(d_in, d_out, kernel_size=3, stride=2, padding=1)
        else:
            raise RuntimeError('Did not specify appropriate convolution type')

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):

        conv_input = enc_input.transpose(1, 2)  # (batch_size, emb_size, n_steps)

        if self.type != 'up':
            conv_output_t = self.conv(conv_input)
        else:
            # we are doing a transpose - we need to specify output size in order to recover the correct size
            output_size = None if non_pad_mask is None else \
                (non_pad_mask.shape[0], conv_input.shape[2], non_pad_mask.shape[1])

            conv_output_t = self.conv(conv_input, output_size=output_size)

        conv_output = conv_output_t.transpose(1, 2)  # (batch_size, n_steps, emb_size)

        # we may not need an activation here, as layer norm can act as an activation

        # if same, we use skip connections from input to output to allow for more efficient gradient propagation
        norm_input = enc_input + conv_output if self.skip_connect else conv_output

        norm_output = self.norm(norm_input)

        # here the output of the convolution performs attention over the input
        #TODO see if using norm output helps
        enc_output, enc_slf_attn = self.slf_attn(
            norm_output, norm_output, norm_output, mask=slf_attn_mask)

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
