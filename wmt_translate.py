import json
import argparse

import youtokentome as yttm
from torch.utis.data import DataLoader

from translation_dataset import RAFTranslationDataset, paired_collate_fn
from transformer.models import Transformer
from transformer.translator import Translator

parser = argparse.ArgumentParser()

parser.add_argument('-config', required=True,
                    help='path to preprocessing config.json')
parser.add_argument('-src', required=True,
                    help='path to text file (one line per sentence)')
parser.add_argument('-beam_size', type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")
parser.add_argument('-no_cuda', action='store_true')


if __name__ == "__main__":
    opt = parser.parse_args()

    opt.cuda = not opt.no_cuda

    with open(opt.config) as f:
        data_config = json.load(f)

    # load tokenizers
    src_bpe = yttm.BPE(model=data_config['src_bpe'])
    tgt_bpe = yttm.BPE(model=data_config['tgt_bpe'])

    # prepare dataloader

    dataset = RAFTranslationDataset(
        src_path=opt.src, tgt_path=None,
        src_tokenizer=lambda x: src_bpe.encode([x], output_type=yttm.OutputType.ID, bos=True, eos=True)[0],
        tgt_tokenizer=lambda x: tgt_bpe.encode([x], output_type=yttm.OutputType.ID, bos=True, eos=True)[0],
        max_len=data_config['max_word_seq_len'],
    )

    dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, n_workers=2, collate_fn=paired_collate_fn, shuffle=False
    )

    # load model

    device = torch.device('cuda' if opt.cuda else 'cpu')
    model_opt = None  # TODO: get model_opt

    model = Transformer(
        model_opt.src_vocab_size,
        model_opt.tgt_vocab_size,
        model_opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_tgt_weight_sharing=model_opt.embs_share_weight,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout,
        unet=model_opt.unet
    ).to(device)

    translator = Translator(opt, model=model)

    # write translated sequences to a file
