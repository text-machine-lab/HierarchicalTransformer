''' Define the Transformer model '''
import torch
from math import sqrt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import transformer.Constants as Constants
from transformer.Layers import UNetEncoderLayer, EncoderLayer, DecoderLayer
from transformer.SubLayers import MultiHeadAttention
from model.utils.vocab import SOS_ID
import wandb

__author__ = "Yu-Hsiang Huang"

def batched_index_select(input, dim, index):
    views = [input.shape[0]] + \
        [1 if i != dim else -1 for i in torch.arange(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)

def get_attn_mask(query_non_pad, key_non_pad):
    '''This calculates the attention mask from purely the non-pad representations'''
    len_q = query_non_pad.size(1)
    key_pad = 1 - key_non_pad
    padding_mask = key_pad.squeeze(2).byte()
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1).bool()  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1
        self.n_layers = n_layers

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.seg_enc = nn.Embedding(len_max_seq, d_word_vec, padding_idx=Constants.PAD)

        # TODO refreeze positional embeddings

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        #self.position_bias = nn.Parameter(torch.ones(1, 1, d_word_vec))

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(self.n_layers)])

    def forward(self, src_seq, src_pos, src_segs=None, return_attns=False):

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq)

        wandb.log({'word_emb_std': enc_output.std()})

        enc_output = enc_output + self.position_enc(src_pos)  # * self.position_bias

        if src_segs is not None:
            enc_output = enc_output + self.seg_enc(src_segs)

        enc_slf_attn_list = []
        for encoder_layer in self.layer_stack:
            enc_output, enc_slf_attn = encoder_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class UNetEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1
        self.n_layers = n_layers

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.seg_enc = nn.Embedding(len_max_seq, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        assert n_layers % 2 == 0  # we have equal up layers as down layers

        self.in_layer = UNetEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, type_='same', skip_connect=True)
        self.out_layer = UNetEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, type_='same', skip_connect=True)

        depth = n_layers // 2 - 1

        # each layer increases in size by the sqrt of 2 to keep computation relatively constant
        # TODO add back sqrt(2) layer sizes
        multiples = [sqrt(2) ** i for i in range(depth + 1)]
        layer_sizes = [round(d_model * multiples[i]) for i in range(depth + 1)]  # [d_model for _ in range(depth+1)]  #
        inner_sizes = [round(d_inner * multiples[i]) for i in range(depth + 1)]
        d_k_sizes = [round(d_k * multiples[i+1]) for i in range(depth)]
        d_v_sizes = [round(d_v * multiples[i+1]) for i in range(depth)]

        # all down layer sizes followed by up layer sizes
        self.d_enc = layer_sizes + layer_sizes[1::-1] + [d_model]

        # layers going down to abstract representation
        #
        self.down_stack = nn.ModuleList([
            UNetEncoderLayer(layer_sizes[i+1], inner_sizes[i+1], n_head, d_k_sizes[i], d_v_sizes[i], dropout=dropout,
                             type_='down', d_in=layer_sizes[i])
            for i in range(depth)])

        layer_sizes.reverse()

        # layers going up to output representation
        self.up_stack = nn.ModuleList([
            UNetEncoderLayer(layer_sizes[i+1], inner_sizes[i+1], n_head, d_k_sizes[i], d_v_sizes[i], dropout=dropout,
                             type_='up', d_in=layer_sizes[i])
            for i in range(depth)])

        self.maxpool1d = nn.MaxPool1d(3, stride=2, padding=1)


    def forward(self, src_seq, src_pos, src_segs=None, return_attns=False):

        # TODO modify this to return a list of hidden layers

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)  # b x l
        non_pad_mask = get_non_pad_mask(src_seq)  # b x lq x lk

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        if src_segs is not None:
            enc_output = enc_output + self.seg_enc(src_segs)

        # start with bit representation of padding tokens (non_pad_mask)

        all_layer_outputs = []
        all_non_pads = []

        slf_attn_list = []
        down_outputs = []

        ######### INPUT LAYER ###########

        enc_output, enc_slf_attn = self.in_layer(
            enc_output,
            non_pad_mask=non_pad_mask,
            slf_attn_mask=slf_attn_mask)

        first_output = enc_output
        all_layer_outputs.append(enc_output)
        all_non_pads.append(non_pad_mask)

        if return_attns:
            slf_attn_list.append(enc_slf_attn)

        ######### DOWN LAYERS ##########

        layer_non_pad = non_pad_mask
        layer_pairs = []
        for layer in self.down_stack:

            prev_layer_non_pad = layer_non_pad  # b x lq
            layer_non_pad = self.maxpool1d(layer_non_pad.transpose(1, 2)).squeeze(1).unsqueeze(2)
            all_non_pads.append(layer_non_pad)

            # compute slf_attn_mask from pad specifications for current and previous layer
            # TODO changing pad mask to go from pooled dim to pooled dim
            # len_q = layer_non_pad.size(1)
            # padding_mask = (1 - layer_non_pad).squeeze(2).byte()
            # padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
            padding_mask = get_attn_mask(layer_non_pad, layer_non_pad)

            enc_output, enc_slf_attn = layer(
                enc_output,
                non_pad_mask=layer_non_pad,
                slf_attn_mask=padding_mask)

            all_layer_outputs.append(enc_output)

            # compute pad specification for next layer from maxpool 1

            # store transposed attention masks for unet decoder

            # TODO removed prev_
            down_outputs.append(enc_output)
            layer_pairs.append((prev_layer_non_pad, layer_non_pad))

            if return_attns:
                slf_attn_list += [enc_slf_attn]

        # we align every up layer with the corresponding down layer
        down_outputs.reverse()
        layer_pairs.reverse()

        enc_output = None  # the first layer of the decoder will not receive input from another decoder layer
        # otherwise, it wouldn't be the first layer

        # decoder uses computed attention masks from unet encoder

        ####### UP LAYERS #############

        for layer, down_output, pair in zip(self.up_stack, down_outputs, layer_pairs):

            # reverse ordering, since we are upsampling now instead of down
            layer_non_pad, prev_layer_non_pad = pair

            all_non_pads.append(layer_non_pad)

            # compute slf_attn_mask from pad specifications for current and previous layer
            len_q = layer_non_pad.size(1)
            #TODO removed prev_
            padding_mask = (1 - layer_non_pad).squeeze(2).byte()
            padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

            # if not first decoder layer, we add input of previous layer with skip connection to respective down layer
            layer_input = enc_output + down_output if enc_output is not None else down_output

            enc_output, enc_slf_attn = layer(
                layer_input,  # HERE WE ADD OUTPUT OF PREVIOUS LAYER WITH SKIP CONNECTION FROM DOWN LAYER
                non_pad_mask=layer_non_pad,
                slf_attn_mask=padding_mask)

            all_layer_outputs.append(enc_output)

            if return_attns:
                slf_attn_list += [enc_slf_attn]

        ######## OUTPUT LAYER #############

        all_non_pads.append(non_pad_mask)

        enc_output, enc_slf_attn = self.out_layer(
            enc_output + first_output,  # HERE WE ADD FIRST DOWN LAYER OUTPUT FOR FINAL PREDICTION
            non_pad_mask=non_pad_mask,
            slf_attn_mask=slf_attn_mask)

        all_layer_outputs.append(enc_output)

        if return_attns:
            slf_attn_list.append(enc_slf_attn)

        if return_attns:
            return enc_output, slf_attn_list

        # TODO now returning all layers outputs, not just the last layer
        # return enc_output,
        all_encoder_layers = list(zip(all_layer_outputs, all_non_pads))

        # TODO allow attention on individual layers
        return enc_output,
        #return all_encoder_layers,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, d_enc=None):

        super().__init__()
        n_position = len_max_seq + 1
        # if none, set encoder size equal to decoder size
        d_enc = d_model if d_enc is None else d_enc
        # if encoder size is an int, expand to a list of ints to initialize each layer respectively
        self.d_enc = [d_enc] * n_layers if isinstance(d_enc, int) else d_enc

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        # TODO here we allow attention over different encoder sizes
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, d_enc=self.d_enc[i])
            for i in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        # TODO modify this so that it can take enc_output as a list of size n_layers
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad.bool() + slf_attn_mask_subseq.bool()).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for i, dec_layer in enumerate(self.layer_stack):
            # here we allow attention over each corresponding encoder layer, instead of just the last
            if isinstance(enc_output, list):
                enc_repr, enc_non_pad = enc_output[i]

                if enc_repr.shape[-1] != self.d_enc[i]:
                    import pdb; pdb.set_trace()

                dec_enc_attn_mask = get_attn_mask(non_pad_mask, enc_non_pad)
            else:
                enc_repr = enc_output

            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_repr,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class GRUEncoder(nn.Module):
    def __init__(self, n_src_vocab, d_model):
        super().__init__()
        self.src_word_emb = nn.Embedding(n_src_vocab, d_model, padding_idx=Constants.PAD)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, src_seq, src_pos=None, src_segs=None):
        non_pad_mask = get_non_pad_mask(src_seq)  # b x t
        src_embs = self.src_word_emb(src_seq)  # b x t x d
        gru_output, _ = self.gru(src_embs)
        #gru_output = gru_output * non_pad_mask

        #import pdb; pdb.set_trace()

        return gru_output,

class MultiHeadAttentionGRUDecoder(nn.Module):
    def __init__(self, n_tgt_vocab, d_model, d_k=64, d_v=64, n_head=8, dropout=0.1):
        super().__init__()

        print('Using GRU!!!!')
        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_model, padding_idx=Constants.PAD)
        self.embedding = self.tgt_word_emb
        self.hidden_enc = nn.Linear(d_model, d_model)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.gru = nn.GRUCell(d_model * 2, d_model)

    def forward(self, tgt_seq, tgt_pos, src_seq, encoder_output, return_attns=False):

        # we look up word embeddings for tgt_seq
        word_input = self.tgt_word_emb(tgt_seq)

        # we calculate attention mask for padding purposes
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # here we grab final states of encoder output

        # TODO restore encoder states as input!
        final_state_idx = (src_seq != 0).sum(1) - 1
        final_states = batched_index_select(encoder_output, 1, final_state_idx).squeeze(1)

        gru_outputs = []
        dec_attns = []
        gru_output = torch.tanh(self.hidden_enc(final_states))  # word_input[:, 0, :].unsqueeze(1)
        for t in range(word_input.shape[1]):
            word_emb = word_input[:, t]
            #word_emb = word_emb # b x d
            step_mask = dec_enc_attn_mask[:, t, :].unsqueeze(1)  #TODO inspect this
            # we perform attention over the encoder
            attn_vec, attn = self.enc_attn(gru_output.unsqueeze(1), encoder_output, encoder_output, mask=step_mask)
            # we concatenate word and attention vector
            gru_input = torch.cat([word_emb, attn_vec.squeeze(1)], 1) # b x 1 x 2d
            # run GRU step to get output
            gru_output = self.gru(gru_input, gru_output)
            # add output to list
            gru_outputs.append(gru_output)
            # save attention maps for viewing
            #dec_attns.append(attn)

        # concatenate all outputs
        outs = torch.stack(gru_outputs, 1)

        #outs, _ = self.gru(word_input, final_states.squeeze(1).unsqueeze(0))

        if return_attns:
            return outs, None, dec_attns  # None because GRU does not do attention over itself
        return outs,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True, unet=True):

        super().__init__()
        self.len_max_seq = len_max_seq
        d_k = d_v = d_model // n_head
        # this is the major modification of the project
        # TODO change back to UNet encoder, right now we are trying GRU decoder
        encoder_type = Encoder # if not unet else UNetEncoder

        self.encoder = encoder_type(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            print('Sharing input/output embeddings of decoder')
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos, src_segs=None, flat_logits=True):
        # TODO add SOS token manually!
        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]
        enc_output, *_ = self.encoder(src_seq, src_pos, src_segs=src_segs)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        if flat_logits:
            return seq_logit.view(-1, seq_logit.size(2))
        else:
            return seq_logit


class MultiModel(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True, encoder='transformer', decoder='transformer'):

        super().__init__()
        self.len_max_seq = len_max_seq
        d_k = d_v = d_model // n_head
        # this is the major modification of the project
        d_enc = d_model

        if encoder == 'transformer':
            self.encoder = Encoder(
                n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                dropout=dropout)

        if encoder == 'unet':
            self.encoder = UNetEncoder(
                n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                dropout=dropout)

            #d_enc = self.encoder.d_enc  # custom layer sizes

        elif encoder == 'gru':
            self.encoder = GRUEncoder(n_src_vocab, d_model)
        else:
            raise RuntimeError('Must specify encoder type')

        if decoder == 'transformer':
            self.decoder = Decoder(
                n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                dropout=dropout, d_enc=d_enc)
        elif decoder == 'gru':
            self.decoder = MultiHeadAttentionGRUDecoder(n_tgt_vocab, d_model, d_k, d_v, n_head, dropout=dropout)
        else:
            raise RuntimeError('Must specify decoder type')

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos, src_segs=None, flat_logits=True):
        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos, src_segs=src_segs)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        if flat_logits:
            return seq_logit.view(-1, seq_logit.size(2))
        else:
            return seq_logit
