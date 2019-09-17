import random
import subprocess
import linecache

import numpy as np
import torch
import torch.utils.data

from transformer import Constants


def paired_collate_fn(insts):
    src_insts, tgt_insts = list(zip(*insts))
    src_insts = collate_fn(src_insts)
    tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, *tgt_insts)


def collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''
    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    batch_pos = np.array([
        [pos_i+1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq])

    batch_seq = torch.LongTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(
        self, src_word2idx, tgt_word2idx,
        src_insts=None, tgt_insts=None):

        assert src_insts
        assert not tgt_insts or (len(src_insts) == len(tgt_insts))

        src_idx2word = {idx:word for word, idx in src_word2idx.items()}
        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word
        self._src_insts = src_insts

        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}
        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word
        self._tgt_insts = tgt_insts

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._tgt_insts:
            return self._src_insts[idx], self._tgt_insts[idx]
        return self._src_insts[idx]


class RAFTranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_path, tgt_path, src_tokenizer, tgt_tokenizer, max_len=None, warm_up=False):
        '''Random access file translation dataset

        reads file using python linecache

        :param warm_up: warm up linecache
        '''

        self.src_path = src_path
        self.tgt_path = tgt_path
        self._len = int(subprocess.check_output("wc -l " + src_path, shell=True).split()[0])
        self._check_data()

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.max_len = max_len

        if warm_up: self[0]  # noqa: E701

    def _check_data(self):
        tgt_len = int(subprocess.check_output("wc -l " + self.tgt_path, shell=True).split()[0])
        if tgt_len != self._len:
            raise RuntimeError(f'different number of line in src ({self._len}) and tgt ({tgt_len}) files')

    def __getitem__(self, idx):
        src_line = linecache.getline(self.src_path, idx + 1)
        tgt_line = linecache.getline(self.tgt_path, idx + 1)

        src_tokens = self.src_tokenizer(src_line.strip())
        tgt_tokens = self.tgt_tokenizer(tgt_line.strip())

        if self.max_len is not None:
            src_tokens = src_tokens[:self.max_len]
            tgt_tokens = tgt_tokens[:self.max_len]

        return src_tokens, tgt_tokens

    def __len__(self):
        return self._len


class StreamingTranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_fpath, tgt_fpath, src_tokenizer, tgt_tokenizer,
                 buffer_size=100_000, shuffle=True, preprocess_fn=None):
        ''' Streaming translation dataset does not store whole dataset in a memory.
            Also, data is preprocessed on-the-fly

            Note: Use it ONLY in Dataloader(shuffle=False),
                  shuffle flag should be specified in this object

            Note: for correct behavior, batch size of the Dataloader should be such
            that buffer_size % batch_size == 0

        :param *_fpath: str, path to the text file with sentences separated with \n
        :param *_tokenizer: function, str -> list of indices
        :param buffer_size: int, read buffer size
        :param shuffle: bool, shuffle buffer
        :param preprocess_fn: function, str -> str, applied before tokenization
        '''

        self.src_fpath = src_fpath
        self.tgt_fpath = tgt_fpath
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.buffer_size = buffer_size
        self.shuffle = shuffle

        self.open_files()
        self._length = None  # lazy

        self._src_buffer = None
        self._tgt_buffer = None

        self._preprocess_fn = preprocess_fn

    def open_files(self):
        self._src_file = open(self.src_fpath)
        self._tgt_file = open(self.tgt_fpath)
        self._index = 0

    def close_files(self):
        self._src_file.close()
        self._tgt_file.close()

    def __del__(self):
        self.close_files()

    def fill_buffers(self):
        src_buffer = []
        tgt_buffer = []
        for _ in range(self.buffer_size):
            self._index += 1

            src_line = self._src_file.readline().strip()
            tgt_line = self._tgt_file.readline().strip()

            if not src_line:
                if tgt_line:
                    raise RuntimeError('Source and target files have different number of sentences')
                self.close_files()
                self.open_files()
                break

            src_buffer.append(src_line)
            tgt_buffer.append(tgt_line)

        if self.shuffle:
            # Note: dataloader should support smaller batch size at the end of the epoch
            random_premutation = np.random.permutation(len(src_buffer))
            src_buffer = [src_buffer[i] for i in random_premutation]
            tgt_buffer = [tgt_buffer[i] for i in random_premutation]

        self._src_buffer = src_buffer
        self._tgt_buffer = tgt_buffer

    def preprocess(self, line, type_):
        '''
        :param type_: 'src' or 'tgt'
        '''
        if type_ not in ('src', 'tgt'):
            raise ValueError('type_ sould be either "src" or "tgt"')

        tokenize = self.src_tokenizer
        if type_ == 'tgt':
            tokenize = self.tgt_tokenizer

        if self._preprocess_fn is not None:
            line = self._preprocess_fn(line)
        return tokenize(line)

    def __len__(self):
        if not self._length:
            with open(self.src_fpath) as f:
                for i, _ in enumerate(f): pass  # noqa: E701
            self._length = i
        return self._length

    def __getitem__(self, idx):
        if idx % self.buffer_size == 0 or self._src_buffer is None:
            # Note: ensure that the whole batch is in one buffer
            # this may be important when accessing elements in parallel
            self.fill_buffers()
            idx = idx % self.buffer_size

        src_line = self._src_buffer[idx]
        tgt_line = self._tgt_buffer[idx]

        src_tokens = self.preprocess(src_line, type_='src')
        tgt_tokens = self.preprocess(tgt_line, type_='tgt')

        return src_tokens, tgt_tokens
