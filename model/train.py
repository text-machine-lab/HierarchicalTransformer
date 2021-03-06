from solver import *
from data_loader import get_loader
from configs import get_config
from utils import Vocab
import os
import pickle
from models import VariationalModels

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    print('Reading configuration')
    config = get_config(mode='train')
    val_config = get_config(mode='valid')
    test_config = get_config(mode='test')
    print(config)
    with open(os.path.join(config.save_path, 'config.txt'), 'w') as f:
        print(config, file=f)

    print('Loading Vocabulary...')
    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')

    config.vocab_size = vocab.vocab_size

    print('Loading training set')
    train_data_loader = get_loader(
        sentences=load_pickle(config.sentences_path),
        conversation_length=load_pickle(config.conversation_length_path),
        sentence_length=load_pickle(config.sentence_length_path),
        vocab=vocab,
        batch_size=config.batch_size,
        max_examples=config.max_examples)

    print('Loading validation set')
    eval_data_loader = get_loader(
        sentences=load_pickle(val_config.sentences_path),
        conversation_length=load_pickle(val_config.conversation_length_path),
        sentence_length=load_pickle(val_config.sentence_length_path),
        vocab=vocab,
        batch_size=val_config.eval_batch_size,
        max_examples=config.max_examples,
        shuffle=False)

    print('Loading test set')
    test_data_loader = get_loader(
        sentences=load_pickle(test_config.sentences_path),
        conversation_length=load_pickle(test_config.conversation_length_path),
        sentence_length=load_pickle(test_config.sentence_length_path),
        vocab=vocab,
        batch_size=test_config.eval_batch_size,
        max_examples=config.max_examples,
        shuffle=False)

    # for testing
    # train_data_loader = eval_data_loader
    if config.model in VariationalModels:
        solver = VariationalSolver
    else:
        solver = Solver

    print('Num train batches: %s' % len(train_data_loader))
    print('Num valid batches: %s' % len(eval_data_loader))

    print('Creating solver')
    solver = solver(config, train_data_loader, eval_data_loader, test_data_loader, vocab=vocab, is_train=True)

    print('Build environment')
    solver.build()
    solver.train()
