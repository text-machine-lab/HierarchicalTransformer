""" Used for large machine translation datasets. Features: full dataset is not stored in memory, BPE tokenization"""
import os
import json
import argparse
import youtokentome as yttm


N_TOKENS = 32_000
VERBOSE = True


def train_bpe(train_src, train_tgt, saveto, vocab_size, verbose=False):
    ''' Creates BPE dictionary and saves it in the saveto directory '''
    src_path = os.path.join(saveto, 'src.bpe')
    tgt_path = os.path.join(saveto, 'tgt.bpe')

    # coverage is 0.9999 according to yttm documentation recommendations
    if verbose: print('[Info] creating BPE dictionary for source language')  # noqa: E701
    yttm.BPE.train(data=train_src, vocab_size=vocab_size, model=src_path, coverage=0.9999)

    if verbose: print('[Info] creating BPE dictionary for target language')  # noqa: E701
    yttm.BPE.train(data=train_tgt, vocab_size=vocab_size, model=tgt_path, coverage=0.9999)

    src_bpe = yttm.BPE(model=src_path)
    tgt_bpe = yttm.BPE(model=tgt_path)
    return src_bpe, tgt_bpe


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-valid_src', required=True)
    parser.add_argument('-valid_tgt', required=True)

    parser.add_argument('-saveto', required=True,
                        help='path to the directory where data will be saved')

    parser.add_argument('-vocab_size', default=32_000)
    # parser.add_argument('-keep_case', action='store_true')
    # parser.add_argument('-max_len', default=50)

    opt = parser.parse_args()

    # script starts here
    src_bpe, tgt_bpe = train_bpe(
        opt.train_src, opt.train_tgt, opt.saveto, opt.vocab_size, verbose=VERBOSE)

    config = vars(opt)
    config['src_bpe'] = os.path.join(config['saveto'], 'src.bpe')
    config['tgt_bpe'] = os.path.join(config['saveto'], 'tgt.bpe')

    config_path = os.path.join(opt.saveto, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f)


if __name__ == "__main__":
    main()
