from itertools import cycle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import models
from layers import masked_cross_entropy
from utils import to_var, time_desc_decorator, pad_and_pack, normal_kl_div, to_bow, bag_of_words_loss, \
    normal_kl_div, embedding_metric, push_zeros_right, SOS_ID
import os
import random
from tqdm import tqdm
from math import isnan
import datetime
#from torch.utils.tensorboard import SummaryWriter
import re
import math
import pickle
import gensim
from models import MULTI

import wandb

import sys
sys.path.append('..')
from transformer.Optim import ScheduledOptim

import tgalert

alert = tgalert.TelegramAlert()


word2vec_path = "../datasets/GoogleNews-vectors-negative300.bin"

def get_gpu_memory_used():
    alloc_mem = 0

    for i in range(torch.cuda.device_count()):
        alloc_mem += torch.cuda.memory_allocated(i)

    return alloc_mem


def extract_history_response(conversations):
    input_histories_ls = [conv[:i] for conv in conversations for i in range(1, len(conv))]
    target_sentences = [conv[i] for conv in conversations for i in range(1, len(conv))]
    input_sentences = [conv[i-1] for conv in conversations for i in range(1, len(conv))]

    input_conversation_length = [len(conversations[i]) - 1 for i in range(len(conversations))]

    input_histories_joined = [[token for sentence in history for token in sentence if token != 0]
                              for history in input_histories_ls]
    max_history_len = max([len(history) for history in input_histories_joined])
    input_histories_padded = [history + [0] * (max_history_len - len(history))
                              for history in input_histories_joined]

    input_history_segs = [[i+1 for i in range(len(history)) for token in history[i] if token != 0]
                              for history in input_histories_ls]
    input_history_segs_padded = [history + [0] * (max_history_len - len(history))
                              for history in input_history_segs]

    # we shuffle in case examples are removed from the end to fit memory requirements
    pairs = list(zip(input_histories_padded, input_history_segs_padded, target_sentences, input_sentences))
    #random.shuffle(pairs)
    # input_histories_shuf = [pair[0] for pair in pairs]
    # input_history_segs_shuf = [pair[1] for pair in pairs]
    # target_sentences_shuf = [pair[2] for pair in pairs]
    # input_sentences_shuf = [pair[3] for pair in pairs]
    # input_conversation_length_shuf = [pair[4] for pair in pairs]
    input_histories_shuf, input_history_segs_shuf, target_sentences_shuf, input_sentences_shuf = zip(*pairs)

    input_histories = to_var(torch.LongTensor(input_histories_shuf))
    target_sentences = to_var(torch.LongTensor(target_sentences_shuf))
    input_sentences = to_var(torch.LongTensor(input_sentences_shuf))
    history_segments = to_var(torch.LongTensor(input_history_segs_shuf))
    conv_lens = to_var(torch.LongTensor(input_conversation_length))

    return input_histories, history_segments, target_sentences, input_sentences, conv_lens


class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class Solver(object):
    def __init__(self, config, train_data_loader, eval_data_loader, test_data_loader, vocab, is_train=True, model=None, parallel=True):
        """This object contains all training and inference code, and receives datasets as input."""
        self.config = config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.test_data_loader = test_data_loader
        self.vocab = vocab
        self.parallel = parallel
        self.is_train = is_train
        self.model = model
        # new code to run Tensorboard
        self.writer = None

    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):
        """Manually called constructor for the Solver object. Initializes the chosen model from models.py, restores
        it from save if specified by the --restore flag and initializes wandb with model configuration details. Also
        sets up the optimizer."""
        if self.model is None:
            self.model = getattr(models, self.config.model)(self.config)

            # orthogonal initialiation for hidden weights
            # input gate bias for GRUs
            if self.config.mode == 'train' and self.config.checkpoint is None:
                print('Parameter initiailization')
                for name, param in self.model.named_parameters():
                    if 'weight_hh' in name:
                        print('\t' + name)
                        nn.init.orthogonal_(param)

                    # bias_hh is concatenation of reset, input, new gates
                    # only set the input gate bias to 2.0
                    if 'bias_hh' in name:
                        print('\t' + name)
                        dim = int(param.size(0) / 3)
                        param.data[dim:2 * dim].fill_(2.0)

        if self.config.restore:
            print('Restoring model from save path')
            self.load_model(self.config.save_path)

        if not self.config.tg_enable:
            alert.disable = True

        n_params = sum([param.numel() for param in self.model.parameters()])
        print('Number of parameters: %s' % n_params)

        str_config = {k: str(v) for k, v in self.config.__dict__.items()}
        wandb.init(project='hierarchical_transformer', notes=self.config.msg, config=str_config)
        wandb.watch(self.model)

        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        # Overview Parameters
        print('Model Parameters')
        for name, param in self.model.named_parameters():
            print('\t' + name + '\t', list(param.size()))

        if self.config.checkpoint:
            self.load_model(self.config.checkpoint)

        if self.is_train:
            # TODO restore scheduled learning rate! or decide you don't want it
            # if isinstance(self.model, TRANSFORMER):
            #
            #     self.optimizer = ScheduledOptim(
            #         optim.Adam(
            #             filter(lambda x: x.requires_grad, self.model.parameters()),
            #             betas=(0.9, 0.98), eps=1e-09),
            #         self.config.encoder_hidden_size, self.config.n_warmup_steps, lr_factor=self.config.learning_rate)
            #
            # else:
            self.optimizer = self.config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate)

    def save_model(self, epoch):
        """Saves model checkpoint in folder to file numbered by epoch, e.g. 1.pkl"""
        """Save parameters to checkpoint"""
        ckpt_path = os.path.join(self.config.save_path, f'{epoch}.pkl')
        print(f'Save parameters to {ckpt_path}')
        torch.save(self.model.state_dict(), ckpt_path)

    def load_model(self, checkpoint):
        """Loads specific checkpoint, if directory is given it will load the checkpoint from the largest epoch."""
        """Load parameters from checkpoint"""

        def fileEpoch(file):
            return int(re.sub('[^0-9]','', file))

        if os.path.isdir(checkpoint):
            files = os.listdir(checkpoint)
            files = [file for file in files if '.pkl' in file]
            files.sort(key=fileEpoch)
            print('Save path is dir. Using largest epoch save: %s' % checkpoint)
            checkpoint = os.path.join(checkpoint, files[-1])

        print(f'Load parameters from {checkpoint}')
        epoch = re.match(r"[0-9]*", os.path.basename(checkpoint)).group(0)
        self.epoch_i = int(epoch)
        self.model.load_state_dict(torch.load(checkpoint))

    # def write_summary(self, epoch_i):
    #     """Not used."""
    #     epoch_loss = getattr(self, 'epoch_loss', None)
    #     if epoch_loss is not None:
    #         self.writer.update_loss(
    #             loss=epoch_loss,
    #             step_i=epoch_i + 1,
    #             name='train_loss')
    #
    #     epoch_recon_loss = getattr(self, 'epoch_recon_loss', None)
    #     if epoch_recon_loss is not None:
    #         self.writer.update_loss(
    #             loss=epoch_recon_loss,
    #             step_i=epoch_i + 1,
    #             name='train_recon_loss')
    #
    #     epoch_kl_div = getattr(self, 'epoch_kl_div', None)
    #     if epoch_kl_div is not None:
    #         self.writer.update_loss(
    #             loss=epoch_kl_div,
    #             step_i=epoch_i + 1,
    #             name='train_kl_div')
    #
    #     kl_mult = getattr(self, 'kl_mult', None)
    #     if kl_mult is not None:
    #         self.writer.update_loss(
    #             loss=kl_mult,
    #             step_i=epoch_i + 1,
    #             name='kl_mult')
    #
    #     epoch_bow_loss = getattr(self, 'epoch_bow_loss', None)
    #     if epoch_bow_loss is not None:
    #         self.writer.update_loss(
    #             loss=epoch_bow_loss,
    #             step_i=epoch_i + 1,
    #             name='bow_loss')
    #
    #     validation_loss = getattr(self, 'validation_loss', None)
    #     if validation_loss is not None:
    #         self.writer.update_loss(
    #             loss=validation_loss,
    #             step_i=epoch_i + 1,
    #             name='validation_loss')

    # def format_convos_for_transformer(self, conversations):
    #     empty = [0] * self.config.max_unroll
    #     max_convo_len = max([len(convo) for convo in conversations])
    #     conversations_pad = [convo + [empty] * (max_convo_len - len(convo)) for convo in conversations]
    #     t_convos = to_var(torch.LongTensor(conversations_pad))
    #     t_sent_lens = torch.sum((t_convos != 0).long(), 2)
    #     # it starts at 1 because the baseline models also don't predict the first response
    #     # for r_idx in range(1, max_convo_len):
    #     return t_convos, t_sent_lens
    #
    #
    # def extract_history_response(self, convos, idx):
    #     # gather flattened conversation history
    #     max_convo_len = convos.shape[1]
    #     histories = convos[:, :idx, :]
    #     histories = histories.view(histories.shape[0], -1)
    #     histories = push_zeros_right(histories)  # move all utterance words to beginning of vector
    #     max_tokens = max_convo_len * self.config.max_unroll
    #     histories = histories[:, :max_tokens]
    #     responses = convos[:, idx, :].contiguous()
    #     histories = to_var(histories)
    #     responses = to_var(responses)
    #
    #     return histories, responses

    @time_desc_decorator('Training Start!')
    def train(self):
        """Trains the HRED, sequence-to-sequence, Transformer and U-Net Transformer models. The 3 latter models are
        combined into one MULTI class contained in models.py."""
        epoch_loss_history = []

        #print('Test before training')
        #self.test()

        #print('\n<Validation before training>...')
        #self.validation_loss = self.evaluate()

        #word_perplexity = self.test()

        min_validation_loss = float('inf')
        min_val_loss_epoch = -1

        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i
            batch_loss_history = []
            n_total_words = 0
            prev_loss = None
            for batch_i, (conversations, conversation_length, sentence_length) in enumerate(tqdm(self.train_data_loader, ncols=80)):
                # conversations: (batch_size) list of conversations
                #   conversation: list of sentences
                #   sentence: list of tokens
                # conversation_length: list of int
                # sentence_length: (batch_size) list of conversation list of sentence_lengths

                for c in range(len(conversations)):
                    assert len(conversations[c]) == conversation_length[c]

                with torch.no_grad():
                    input_histories, history_segments, target_sentences, input_sentences, input_conversation_length \
                        = extract_history_response(conversations)

                    target_sentence_length = (target_sentences != 0).long().sum(1)
                    input_sentence_length = (input_sentences != 0).long().sum(1)

                    # MAKE SURE THAT EVALUATION FUNCTION MATCHES THESE RESTRICTIONS ON INPUT SIZE!!!

                    wandb.log({'hist_max_len': (input_histories != 0).float().sum(1).max()})

                    # TODO right align input histories and prune from left

                    input_histories = input_histories[:, :self.config.max_history]
                    history_segments = history_segments[:, :self.config.max_history]

                self.model.train()
                self.optimizer.zero_grad()


                if isinstance(self.model, MULTI):
                    sentence_logits = self.model(input_histories, history_segments, target_sentences, decode=False)
                else:
                    sentence_logits = self.model(input_sentences, input_sentence_length, input_conversation_length,
                                                 target_sentences, decode=False)

                batch_loss, n_words = masked_cross_entropy(
                    sentence_logits,
                    target_sentences,
                    target_sentence_length)

                wandb.log({'memory_used': get_gpu_memory_used()})

                # Back-propagation
                batch_loss.backward()

                if isinstance(self.model, MULTI):
                    wandb.log({'word_grad': self.model.model.encoder.src_word_emb.weight.grad.abs().mean()})

                # Gradient cliping
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

                #self.writer.add_scalar('grad_norm', norm, tb_idx)
                wandb.log({'grad_norm': norm})

                # Run optimizer
                self.optimizer.step()

                assert not isnan(batch_loss.item())
                batch_loss_history.append(batch_loss.item())
                n_total_words += n_words.item()

                current_loss = batch_loss.item() / n_words
                #if prev_loss is None: prev_loss = current_loss
                #self.writer.add_scalar('training_loss', current_loss, tb_idx)
                #self.writer.add_scalar('Training_loss_change', current_loss - prev_loss, tb_idx)
                #prev_loss = current_loss

                wandb.log({'train_loss': current_loss})

                # if batch_i % self.config.print_every == 0:
                #     tqdm.write(
                #         f'Epoch: {epoch_i+1}, iter {batch_i}: loss = {batch_loss.item()/ n_words.item():.3f}')

            epoch_loss = np.sum(batch_loss_history) / n_total_words
            epoch_loss_history.append(epoch_loss)
            self.epoch_loss = epoch_loss

            print_str = f'Epoch {epoch_i+1} loss average: {epoch_loss:.3f}'
            print(print_str)

            print('\n<Validation>...')
            self.validation_loss = self.evaluate()
            if self.validation_loss <= min_validation_loss:
                min_validation_loss = self.validation_loss
                min_val_loss_epoch = epoch_i

            if min_validation_loss == self.validation_loss:
                print('Lowest validation loss yet. Saving')
                self.save_model(epoch_i + 1)
            else:
                print('Validation loss increased. Stopping training')
                break

            #if epoch_i % self.config.plot_every_epoch == 0:
                    #self.write_summary(epoch_i)

        #self.save_model(self.config.n_epoch)

        if self.config.n_epoch > 0:
            print('Loading model from lowest validation epoch')
            ckpt_path = os.path.join(self.config.save_path, f'{min_val_loss_epoch + 1}.pkl')
            self.load_model(ckpt_path)

        wandb.config.update({'min_val_loss': min_validation_loss})

        alert.write('solver.py: Finished training')
        print('Lowest validation loss: %s' % min_validation_loss)

        print('Evaluating on test set')
        word_perplexity = self.test()

        wandb.config.update({'test perplexity': word_perplexity})

        return epoch_loss_history

    def generate_sentence(self, input_sentences, input_histories, input_sentence_length,
                          input_conversation_length, target_sentences, file=None, verbose=True):
        """Generates a sentence from the HRED model only."""
        self.model.eval()

        # save to a text file
        if file is None:
            file = os.path.join(self.config.save_path, 'samples.txt')

        # [batch_size, max_seq_len, vocab_size]
        generated_sentences = self.model(
            input_sentences,
            input_sentence_length,
            input_conversation_length,
            target_sentences,
            decode=True)

        # write output to file
        with open(file, 'a') as f:
            f.write(f'<Epoch {self.epoch_i}>\n\n')

            if verbose: tqdm.write('\n<Samples>')
            for input_sent, target_sent, output_sent in zip(input_histories, target_sentences, generated_sentences):
                input_sent = self.vocab.decode(input_sent, stop_at_eos=False)
                target_sent = self.vocab.decode(target_sent)
                output_sent = '\n'.join([self.vocab.decode(sent) for sent in output_sent])
                s = '\n'.join(['Input sentence: ' + input_sent,
                               'Ground truth: ' + target_sent,
                               'Generated response: ' + output_sent + '\n'])
                f.write(s + '\n')
                if verbose: print(s)
            if verbose: print('')


    def generate_transformer_sentence(self, input_histories, history_segments, target_sentences, file=None, verbose=True):
        """Generate a sentence from the sequence-to-sequence (S2SA), Transformer or U-Net Transformer models."""
        self.model.eval()

        #gold = self.add_sos(target_sentences)

        if file is None:
            file = os.path.join(self.config.save_path, 'samples.txt')

        # [batch_size, max_seq_len, vocab_size]
        generated_sentences = self.model(input_histories, history_segments, target_sentences, decode=True)

        generated_sentences = [sentence for sentence in generated_sentences]

        # write output to file
        with open(file, 'a') as f:
            f.write(f'<Epoch {self.epoch_i}>\n\n')

            if verbose: tqdm.write('\n<Samples>')
            for input_sent, target_sent, output_sent in zip(input_histories, target_sentences, generated_sentences):
                input_sent = self.vocab.decode(input_sent, stop_at_eos=False)
                target_sent = self.vocab.decode(target_sent)
                output_sent = self.vocab.decode(output_sent)
                #output_sent = '\n'.join([self.vocab.decode(sent) for sent in output_sent])
                s = '\n'.join(['Input sentence: ' + input_sent,
                               'Ground truth: ' + target_sent,
                               'Generated response: ' + output_sent + '\n'])
                f.write(s + '\n')

                if verbose: print(s)
            if verbose: print('')


    def evaluate(self):
        """Calculate cross entropy on the validation set, also print out generated sentences. This runs once per
        epoch."""
        self.model.eval()
        batch_loss_history = []
        n_total_words = 0

        for batch_i, (conversations, conversation_length, sentence_length) in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            # conversations: (batch_size) list of conversations
            #   conversation: list of sentences
            #   sentence: list of tokens
            # conversation_length: list of int
            # sentence_length: (batch_size) list of conversation list of sentence_lengths

            with torch.no_grad():

                input_histories, history_segments, target_sentences, input_sentences, input_conversation_length \
                    = extract_history_response(conversations)

                target_sentence_length = (target_sentences != 0).long().sum(1)
                input_sentence_length = (input_sentences != 0).long().sum(1)

                #if isinstance(self.model, TRANSFORMER):

                    # this can protect against random memory shortages
                input_histories = input_histories[:, :self.config.max_history]
                history_segments = history_segments[:, :self.config.max_history]
                    # target_sentences = target_sentences[:self.config.max_convo_len, :self.config.max_unroll]

                    #gold = self.add_sos(target_sentences)

                if isinstance(self.model, MULTI):
                    sentence_logits = self.model(input_histories, history_segments, target_sentences, decode=False)
                else:
                    sentence_logits = self.model(input_sentences, input_sentence_length, input_conversation_length,
                                                 target_sentences, decode=False)

                    #batch_loss, n_words = masked_cross_entropy(sentence_logits, target_sentences, sentence_lens)

                if batch_i == 0:
                    if isinstance(self.model, MULTI):
                        self.generate_transformer_sentence(input_histories, history_segments, target_sentences)
                    else:
                        self.generate_sentence(input_sentences, input_histories, input_sentence_length, input_conversation_length, target_sentences)


                batch_loss, n_words = masked_cross_entropy(
                    sentence_logits,
                    target_sentences,
                    target_sentence_length)

            assert not isnan(batch_loss.item())
            batch_loss_history.append(batch_loss.item())
            n_total_words += n_words.item()

        epoch_loss = np.sum(batch_loss_history) / n_total_words

        #self.writer.add_scalar('val_loss', epoch_loss, self.epoch_i)
        wandb.log({'val_loss': epoch_loss})

        print_str = f'Validation loss: {epoch_loss:.3f}\n'
        print(print_str)

        return epoch_loss

    def test(self):
        """Evaluates per-word perplexity."""
        self.model.eval()
        batch_loss_history = []
        n_total_words = 0

        if self.config.full_samples_file is not None:
            open(self.config.full_samples_file, 'w').close()  # clear all contents of file

        for batch_i, (conversations, conversation_length, sentence_length) in enumerate(tqdm(self.test_data_loader, ncols=80)):
            # conversations: (batch_size) list of conversations
            #   conversation: list of sentences
            #   sentence: list of tokens
            # conversation_length: list of int
            # sentence_length: (batch_size) list of conversation list of sentence_lengths

            with torch.no_grad():

                input_histories, history_segments, target_sentences, input_sentences, input_conversation_length \
                    = extract_history_response(conversations)

                target_sentence_length = (target_sentences != 0).long().sum(1)
                input_sentence_length = (input_sentences != 0).long().sum(1)

                input_histories = input_histories[:, :self.config.max_history]
                history_segments = history_segments[:, :self.config.max_history]

                #if isinstance(self.model, TRANSFORMER):

                    # this can protect against random memory shortages
                    # input_histories = input_histories[:self.config.max_convo_len, :self.config.max_unroll]
                    # history_segments = history_segments[:self.config.max_convo_len, :self.config.max_unroll]
                    # target_sentences = target_sentences[:self.config.max_convo_len, :self.config.max_unroll]

                    #gold = self.add_sos(target_sentences)

                if isinstance(self.model, MULTI):
                    sentence_logits = self.model(input_histories, history_segments, target_sentences, decode=False)
                else:
                    sentence_logits = self.model(input_sentences, input_sentence_length, input_conversation_length,
                                                 target_sentences, decode=False)


                if self.config.full_samples_file is not None:

                    if self.config.max_samples is None or self.config.max_samples >= batch_i * self.config.batch_size:

                        if isinstance(self.model, MULTI):
                            self.generate_transformer_sentence(input_histories, history_segments, target_sentences,
                                                               file=self.config.full_samples_file, verbose=False)
                        else:
                            self.generate_sentence(input_sentences, input_histories, input_sentence_length, input_conversation_length, target_sentences,
                                                   file=self.config.full_samples_file, verbose=False)

                    #batch_loss, n_words = masked_cross_entropy(sentence_logits, target_sentences, sentence_lens)

                # TODO allow generation from HRED
                #if batch_i == 0:
                #    self.generate_transformer_sentence(input_histories, history_segments, target_sentences)


                batch_loss, n_words = masked_cross_entropy(
                    sentence_logits,
                    target_sentences,
                    target_sentence_length)

            assert not isnan(batch_loss.item())
            batch_loss_history.append(batch_loss.item())
            n_total_words += n_words.item()

        epoch_loss = np.sum(batch_loss_history) / n_total_words

        print(f'Number of words: {n_total_words}')
        print(f'Bits per word: {epoch_loss:.3f}')
        word_perplexity = np.exp(epoch_loss)

        print_str = f'Word perplexity : {word_perplexity:.3f}\n'
        print(print_str)

        return word_perplexity

    def embedding_metric(self):
        word2vec =  getattr(self, 'word2vec', None)
        if word2vec is None:
            print('Loading word2vec model')
            word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
            self.word2vec = word2vec
        keys = word2vec.vocab
        self.model.eval()
        n_context = self.config.n_context
        n_sample_step = self.config.n_sample_step
        metric_average_history = []
        metric_extrema_history = []
        metric_greedy_history = []
        context_history = []
        sample_history = []
        n_sent = 0
        n_conv = 0
        for batch_i, (conversations, conversation_length, sentence_length) \
                in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            # conversations: (batch_size) list of conversations
            #   conversation: list of sentences
            #   sentence: list of tokens
            # conversation_length: list of int
            # sentence_length: (batch_size) list of conversation list of sentence_lengths

            conv_indices = [i for i in range(len(conversations)) if len(conversations[i]) >= n_context + n_sample_step]
            context = [c for i in conv_indices for c in [conversations[i][:n_context]]]
            ground_truth = [c for i in conv_indices for c in [conversations[i][n_context:n_context + n_sample_step]]]
            sentence_length = [c for i in conv_indices for c in [sentence_length[i][:n_context]]]

            with torch.no_grad():
                context = to_var(torch.LongTensor(context))
                sentence_length = to_var(torch.LongTensor(sentence_length))

            samples = self.model.generate(context, sentence_length, n_context)

            context = context.data.cpu().numpy().tolist()
            samples = samples.data.cpu().numpy().tolist()
            context_history.append(context)
            sample_history.append(samples)

            samples = [[self.vocab.decode(sent) for sent in c] for c in samples]
            ground_truth = [[self.vocab.decode(sent) for sent in c] for c in ground_truth]

            samples = [sent for c in samples for sent in c]
            ground_truth = [sent for c in ground_truth for sent in c]

            samples = [[word2vec[s] for s in sent.split() if s in keys] for sent in samples]
            ground_truth = [[word2vec[s] for s in sent.split() if s in keys] for sent in ground_truth]

            indices = [i for i, s, g in zip(range(len(samples)), samples, ground_truth) if s != [] and g != []]
            samples = [samples[i] for i in indices]
            ground_truth = [ground_truth[i] for i in indices]
            n = len(samples)
            n_sent += n

            metric_average = embedding_metric(samples, ground_truth, word2vec, 'average')
            metric_extrema = embedding_metric(samples, ground_truth, word2vec, 'extrema')
            metric_greedy = embedding_metric(samples, ground_truth, word2vec, 'greedy')
            metric_average_history.append(metric_average)
            metric_extrema_history.append(metric_extrema)
            metric_greedy_history.append(metric_greedy)

        epoch_average = np.mean(np.concatenate(metric_average_history), axis=0)
        epoch_extrema = np.mean(np.concatenate(metric_extrema_history), axis=0)
        epoch_greedy = np.mean(np.concatenate(metric_greedy_history), axis=0)

        print('n_sentences:', n_sent)
        print_str = f'Metrics - Average: {epoch_average:.3f}, Extrema: {epoch_extrema:.3f}, Greedy: {epoch_greedy:.3f}'
        print(print_str)
        print('\n')

        return epoch_average, epoch_extrema, epoch_greedy


class VariationalSolver(Solver):

    def __init__(self, config, train_data_loader, eval_data_loader, test_data_loader, vocab, is_train=True, model=None):
        self.config = config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.test_data_loader = test_data_loader
        self.vocab = vocab
        self.is_train = is_train
        self.model = model

    @time_desc_decorator('Training Start!')
    def train(self):

        #TODO do not modify this!!!

        epoch_loss_history = []
        kl_mult = 0.0
        conv_kl_mult = 0.0
        min_val_loss = float('inf')
        min_val_loss_epoch = 0
        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i
            batch_loss_history = []
            recon_loss_history = []
            kl_div_history = []
            kl_div_sent_history = []
            kl_div_conv_history = []
            bow_loss_history = []
            self.model.train()
            n_total_words = 0

            # self.evaluate()

            for batch_i, (conversations, conversation_length, sentence_length) \
                    in enumerate(tqdm(self.train_data_loader, ncols=80)):
                # conversations: (batch_size) list of conversations
                #   conversation: list of sentences
                #   sentence: list of tokens
                # conversation_length: list of int
                # sentence_length: (batch_size) list of conversation list of sentence_lengths

                target_conversations = [conv[1:] for conv in conversations]

                # flatten input and target conversations
                sentences = [sent for conv in conversations for sent in conv]
                input_conversation_length = [l - 1 for l in conversation_length]
                target_sentences = [sent for conv in target_conversations for sent in conv]
                target_sentence_length = [l for len_list in sentence_length for l in len_list[1:]]
                sentence_length = [l for len_list in sentence_length for l in len_list]

                sentences = to_var(torch.LongTensor(sentences))
                sentence_length = to_var(torch.LongTensor(sentence_length))
                input_conversation_length = to_var(torch.LongTensor(input_conversation_length))
                target_sentences = to_var(torch.LongTensor(target_sentences))
                target_sentence_length = to_var(torch.LongTensor(target_sentence_length))
                #
                # target_sentence_length = (target_sentences != 0).long().sum(1)
                # input_sentence_length = (input_sentences != 0).long().sum(1)

                # reset gradient
                self.optimizer.zero_grad()

                sentence_logits, kl_div, _, _ = self.model(
                    sentences,
                    sentence_length,
                    input_conversation_length,
                    target_sentences)

                recon_loss, n_words = masked_cross_entropy(
                    sentence_logits,
                    target_sentences,
                    target_sentence_length)

                per_word_loss = recon_loss / n_words

                wandb.log({'train_loss': per_word_loss})

                batch_loss = recon_loss + kl_mult * kl_div
                batch_loss_history.append(batch_loss.item())
                recon_loss_history.append(recon_loss.item())
                kl_div_history.append(kl_div.item())
                n_total_words += n_words.item()

                if self.config.bow:
                    bow_loss = self.model.compute_bow_loss(target_conversations)
                    batch_loss += bow_loss
                    bow_loss_history.append(bow_loss.item())

                assert not isnan(batch_loss.item())

                if batch_i % self.config.print_every == 0:
                    print_str = f'Epoch: {epoch_i+1}, iter {batch_i}: loss = {batch_loss.item() / n_words.item():.3f}, recon = {recon_loss.item() / n_words.item():.3f}, kl_div = {kl_div.item() / n_words.item():.3f}'
                    if self.config.bow:
                        print_str += f', bow_loss = {bow_loss.item() / n_words.item():.3f}'
                    tqdm.write(print_str)

                # Back-propagation
                batch_loss.backward()

                # Gradient cliping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

                # Run optimizer
                self.optimizer.step()
                kl_mult = min(kl_mult + 1.0 / self.config.kl_annealing_iter, 1.0)

            epoch_loss = np.sum(batch_loss_history) / n_total_words
            epoch_loss_history.append(epoch_loss)

            epoch_recon_loss = np.sum(recon_loss_history) / n_total_words
            epoch_kl_div = np.sum(kl_div_history) / n_total_words

            self.kl_mult = kl_mult
            self.epoch_loss = epoch_loss
            self.epoch_recon_loss = epoch_recon_loss
            self.epoch_kl_div = epoch_kl_div

            print_str = f'Epoch {epoch_i+1} loss average: {epoch_loss:.3f}, recon_loss: {epoch_recon_loss:.3f}, kl_div: {epoch_kl_div:.3f}'
            if bow_loss_history:
                self.epoch_bow_loss = np.sum(bow_loss_history) / n_total_words
                print_str += f', bow_loss = {self.epoch_bow_loss:.3f}'
            print(print_str)

            print('\n<Validation>...')
            self.validation_loss = self.evaluate()
            if self.validation_loss <= min_val_loss:
                min_val_loss = self.validation_loss
                min_val_loss_epoch = epoch_i
                self.save_model(epoch_i + 1)
            else:
                print('Validation loss increased. Stop training')
                break

            #if epoch_i % self.config.plot_every_epoch == 0:
                    #self.write_summary(epoch_i)

        if self.config.n_epoch > 0:
            print('Loading model from lowest validation epoch')
            ckpt_path = os.path.join(self.config.save_path, f'{min_val_loss_epoch + 1}.pkl')
            self.load_model(ckpt_path)


        print('Evaluating test perplexity')
        word_perplexity = self.importance_sample(eval=False)

        wandb.config.update({'min_val_loss': min_val_loss})

        wandb.config.update({'test perplexity': word_perplexity})

        return epoch_loss_history

    def generate_sentence(self, sentences, input_histories, sentence_length,
                          input_conversation_length, input_sentences, target_sentences, file=None, verbose=True):
        """Generate output of decoder (single batch)"""

        self.model.eval()

        if file is None:
            file = os.path.join(self.config.save_path, 'samples.txt')

        # [batch_size, max_seq_len, vocab_size]
        generated_sentences, _, _, _ = self.model(
            sentences,
            sentence_length,
            input_conversation_length,
            target_sentences,
            decode=True)

        # write output to file
        with open(file, 'a') as f:
            f.write(f'<Epoch {self.epoch_i}>\n\n')

            if verbose: tqdm.write('\n<Samples>')
            for input_sent, target_sent, output_sent in zip(input_histories, target_sentences, generated_sentences):
                input_sent = self.vocab.decode(input_sent, stop_at_eos=False)
                target_sent = self.vocab.decode(target_sent)
                output_sent = '\n'.join([self.vocab.decode(sent) for sent in output_sent])
                s = '\n'.join(['Input sentence: ' + input_sent,
                               'Ground truth: ' + target_sent,
                               'Generated response: ' + output_sent + '\n'])
                f.write(s + '\n')
                if verbose: print(s)
            if verbose: print('')

    def evaluate(self):
        self.model.eval()
        batch_loss_history = []
        recon_loss_history = []
        kl_div_history = []
        bow_loss_history = []
        n_total_words = 0
        for batch_i, (conversations, conversation_length, sentence_length) \
                in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            # conversations: (batch_size) list of conversations
            #   conversation: list of sentences
            #   sentence: list of tokens
            # conversation_length: list of int
            # sentence_length: (batch_size) list of conversation list of sentence_lengths

            #TODO do not modify this!!!

            target_conversations = [conv[1:] for conv in conversations]

            # flatten input and target conversations
            sentences = [sent for conv in conversations for sent in conv]
            input_conversation_length = [l - 1 for l in conversation_length]
            target_sentences = [sent for conv in target_conversations for sent in conv]
            target_sentence_length = [l for len_list in sentence_length for l in len_list[1:]]
            sentence_length = [l for len_list in sentence_length for l in len_list]

            with torch.no_grad():
                sentences = to_var(torch.LongTensor(sentences))
                sentence_length = to_var(torch.LongTensor(sentence_length))
                input_conversation_length = to_var(
                    torch.LongTensor(input_conversation_length))
                target_sentences = to_var(torch.LongTensor(target_sentences))
                target_sentence_length = to_var(torch.LongTensor(target_sentence_length))

            if batch_i == 0:
                input_histories, history_segments, _, _, _ \
                    = extract_history_response(conversations)
                input_conversations = [conv[:-1] for conv in conversations]
                input_sentences = [sent for conv in input_conversations for sent in conv]
                with torch.no_grad():
                    input_sentences = to_var(torch.LongTensor(input_sentences))
                self.generate_sentence(sentences,
                                       input_histories,
                                       sentence_length,
                                       input_conversation_length,
                                       input_sentences,
                                       target_sentences)

            sentence_logits, kl_div, _, _ = self.model(
                sentences,
                sentence_length,
                input_conversation_length,
                target_sentences)

            recon_loss, n_words = masked_cross_entropy(
                sentence_logits,
                target_sentences,
                target_sentence_length)

            batch_loss = recon_loss + kl_div
            if self.config.bow:
                bow_loss = self.model.compute_bow_loss(target_conversations)
                bow_loss_history.append(bow_loss.item())

            assert not isnan(batch_loss.item())
            batch_loss_history.append(batch_loss.item())
            recon_loss_history.append(recon_loss.item())
            kl_div_history.append(kl_div.item())
            n_total_words += n_words.item()

        epoch_loss = np.sum(batch_loss_history) / n_total_words
        epoch_recon_loss = np.sum(recon_loss_history) / n_total_words
        epoch_kl_div = np.sum(kl_div_history) / n_total_words

        print_str = f'Validation loss: {epoch_loss:.3f}, recon_loss: {epoch_recon_loss:.3f}, kl_div: {epoch_kl_div:.3f}'
        if bow_loss_history:
            epoch_bow_loss = np.sum(bow_loss_history) / n_total_words
            print_str += f', bow_loss = {epoch_bow_loss:.3f}'
        print(print_str)
        print('\n')

        return epoch_loss

    def importance_sample(self, eval=True):
        ''' Perform importance sampling to get tighter bound
        '''

        open(self.config.full_samples_file, 'w').close()  # clear samples file

        loader = self.eval_data_loader if eval else self.test_data_loader

        self.model.eval()
        weight_history = []
        n_total_words = 0
        kl_div_history = []
        for batch_i, (conversations, conversation_length, sentence_length) \
                in enumerate(tqdm(loader, ncols=80)):
            # conversations: (batch_size) list of conversations
            #   conversation: list of sentences
            #   sentence: list of tokens
            # conversation_length: list of int
            # sentence_length: (batch_size) list of conversation list of sentence_lengths

            target_conversations = [conv[1:] for conv in conversations]

            # flatten input and target conversations
            sentences = [sent for conv in conversations for sent in conv]
            input_conversation_length = [l - 1 for l in conversation_length]
            target_sentences = [sent for conv in target_conversations for sent in conv]
            target_sentence_length = [l for len_list in sentence_length for l in len_list[1:]]
            sentence_length = [l for len_list in sentence_length for l in len_list]

            # n_words += sum([len([word for word in sent if word != PAD_ID]) for sent in target_sentences])
            with torch.no_grad():
                sentences = to_var(torch.LongTensor(sentences))
                sentence_length = to_var(torch.LongTensor(sentence_length))
                input_conversation_length = to_var(
                    torch.LongTensor(input_conversation_length))
                target_sentences = to_var(torch.LongTensor(target_sentences))
                target_sentence_length = to_var(torch.LongTensor(target_sentence_length))

                # TODO figure out if this prints to file
                if self.config.full_samples_file is not None:
                    input_histories, history_segments, _, _, _ \
                        = extract_history_response(conversations)
                    if self.config.max_samples is None or self.config.max_samples >= batch_i * self.config.batch_size:
                        input_conversations = [conv[:-1] for conv in conversations]
                        input_sentences = [sent for conv in input_conversations for sent in conv]
                        self.generate_sentence(sentences, input_histories, sentence_length, input_conversation_length,
                                               input_sentences, target_sentences, file=self.config.full_samples_file, verbose=False)

            # treat whole batch as one data sample
            weights = []
            for j in range(self.config.importance_sample):
                sentence_logits, kl_div, log_p_z, log_q_zx = self.model(
                    sentences,
                    sentence_length,
                    input_conversation_length,
                    target_sentences)

                recon_loss, n_words = masked_cross_entropy(
                    sentence_logits,
                    target_sentences,
                    target_sentence_length)

                log_w = (-recon_loss.sum() + log_p_z - log_q_zx).data
                weights.append(log_w)
                if j == 0:
                    n_total_words += n_words.item()
                    kl_div_history.append(kl_div.item())

            # weights: [n_samples]
            weights = torch.stack(weights, 0)
            m = np.floor(weights.max().item())
            weights = np.log(torch.exp(weights - m).sum().item())
            weights = m + weights - np.log(self.config.importance_sample)
            weight_history.append(weights)

        print(f'Number of words: {n_total_words}')
        bits_per_word = -np.sum(weight_history) / n_total_words
        print(f'Bits per word: {bits_per_word:.3f}')
        word_perplexity = np.exp(bits_per_word)

        epoch_kl_div = np.sum(kl_div_history) / n_total_words

        print_str = f'Word perplexity upperbound using {self.config.importance_sample} importance samples: {word_perplexity:.3f}, kl_div: {epoch_kl_div:.3f}\n'
        print(print_str)

        return word_perplexity

