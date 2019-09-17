import os
import sys
import random
import unittest

sys.path.append('..')
from translation_dataset import StreamingTranslationDataset, paired_collate_fn  # noqa: E402


VERBOSE = True


def mock_tokenizer(text):
    return [ord(token[0]) for token in text.split()]


class TestStreamingTranslationDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open('test_src.txt', 'w') as f:
            for _ in range(1000):
                print("".join([random.choice("abcd ") for _ in range(100)]), file=f)

        with open('test_tgt.txt', 'w') as f:
            for _ in range(1000):
                print("".join([random.choice("abcd ") for _ in range(100)]), file=f)

        with open('test_tgt_fail.txt', 'w') as f:
            for _ in range(1000):
                print("".join([random.choice("abcd ") for _ in range(100)]), file=f)

    @classmethod
    def tearDownClass(cls):
        os.remove('test_src.txt')
        os.remove('test_tgt.txt')
        os.remove('test_tgt_fail.txt')

    def test_creation(self):
        _ = StreamingTranslationDataset('test_src.txt', 'test_tgt.txt',
                                        mock_tokenizer, mock_tokenizer)

    def test_getitem(self):
        ds = StreamingTranslationDataset('test_src.txt', 'test_tgt.txt',
                                         mock_tokenizer, mock_tokenizer)
        res = ds[0]
        self.assertEqual(len(res), 2)

        if VERBOSE:
            print('src: ', res[0])
            print('tgt: ', res[0])

        res = ds[1]
        self.assertEqual(len(res), 2)

    def test_len(self):
        ds = StreamingTranslationDataset('test_src.txt', 'test_tgt.txt',
                                         mock_tokenizer, mock_tokenizer)
        length = len(ds)
        self.assertTrue(isinstance(length, int))

    def test_dataloader(self):
        from torch.utils.data import DataLoader
        ds = StreamingTranslationDataset('test_src.txt', 'test_tgt.txt',
                                         mock_tokenizer, mock_tokenizer)
        dl = DataLoader(ds, batch_size=16, shuffle=False)
        for _ in dl:
            pass

    def test_dataloader_workers(self):
        from torch.utils.data import DataLoader
        ds = StreamingTranslationDataset('test_src.txt', 'test_tgt.txt',
                                         mock_tokenizer, mock_tokenizer)
        dl = DataLoader(ds, batch_size=16, num_workers=2, shuffle=False)
        for _ in dl:
            pass

        # ds = StreamingTranslationDataset('test_src.txt', 'test_tgt.txt',
        #                                  mock_tokenizer, mock_tokenizer)
        # dl = DataLoader(ds, batch_size=16, num_workers=2, collate_fn=paired_collate_fn, shuffle=False)
        # for _ in dl:
        #     pass


if __name__ == "__main__":
    unittest.main()
