'''
This script handling the training process for large translation dataset
'''

import os
import argparse
import math
import time
import json

import wandb
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.distributed as distr
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import youtokentome as yttm
from sacrebleu import corpus_bleu

from tqdm import tqdm

import transformer.Constants as Constants
from translation_dataset import RAFTranslationDataset, paired_collate_fn
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from transformer.Translator import Translator
from transformer.Utils import WrappedDistributedDataParallel

SPECIAL_TOKENS = {Constants.PAD, Constants.UNK, Constants.BOS, Constants.EOS}


def detorch(x):
    return x.detach().cpu().numpy()


def perplexity(x):
    return math.exp(min(x, 100))


def cal_performance(pred, gold, smoothing=False):
    ''' Calculate loss and the number of correctly predicted tokens,
    apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def compute_bleu(model, dataloader, beam_size, max_seq_len):
    if dataloader.dataset.tgt_detokenizer is None:
        raise RuntimeError('tgt_detokenizer should be specified in the dataset '
                           'in order to compute BLEU score')

    detokenize = dataloader.dataset.tgt_detokenizer

    pred_translations = []
    true_translations = []

    translator = Translator(None, model, beam_size=beam_size, max_seq_len=max_seq_len, n_best=1)
    for src_seq, src_pos, tgt_seq, tgt_pos in tqdm(
            dataloader, mininterval=2, desc='  - (Translating)', leave=True):
        all_hyp, all_scores = translator.translate_batch(src_seq, src_pos)

        for pred_sentence_hyp, true_translation in zip(all_hyp, tgt_seq):
            # iteration over a batch dimension

            translation_ids = pred_sentence_hyp[0]  # select the first hypothesis
            true_translation_ids = detorch(true_translation).tolist()

            translation_ids = [t for t in translation_ids if t not in SPECIAL_TOKENS]
            true_translation_ids = [t for t in true_translation_ids if t not in SPECIAL_TOKENS]

            translation_str = true_translation_str = ''
            if translation_ids:
                translation_str = detokenize(translation_ids)
            if true_translation_ids:
                true_translation_str = detokenize(true_translation_ids)

            # standard tokenization for WMT dataset BLEU computation
            pred_translations.append(translation_str)
            true_translations.append(true_translation_str)

    bleu_obj = corpus_bleu(pred_translations, [true_translations])
    return bleu_obj.score


def eval_epoch(model, validation_data, device, beam_size=None, max_seq_len=None):
    ''' Epoch operation in evaluation phase

    :param beam_size: int, needed for BLEU score computation
    :param max_seq_len: int, needed for BLEU score computation
    '''
    start = time.time()
    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data,
                desc='  - (Validation) ', leave=True):

            # prepare data
            # pos - position, it is added in collate_fn
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

        if beam_size is not None:
            valid_bleu = compute_bleu(model, validation_data, beam_size, max_seq_len)
            wandb.log({'val_bleu': valid_bleu})

    loss_per_word = total_loss/n_word_total

    wandb.log({'val_loss': loss_per_word})

    accuracy = n_word_correct/n_word_total

    print(f'  - (Validation) ppl: {round(perplexity(loss_per_word), 5)}, '
          f'accuracy: {round(accuracy, 3)} %, '
          f'elapse: {round((time.time()-start)/60, 3)} min')

    model.train()
    return loss_per_word, accuracy, valid_bleu


def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Train and evaluate the model

    :param training_data: torch DataLoader
    :param opt: argparse.Namespace
    '''

    global_step = 0
    max_bleu = 0

    for epoch_i in range(opt.epoch):
        print(f'[ Epoch {epoch_i} ]')
        start = time.time()

        for batch_idx, batch in enumerate(tqdm(training_data, desc='  - (Training)   ', leave=True)):
            global_step += 1

            # evaluate
            if global_step % opt.eval_every == 0:
                print('    - [Info] in-epoch evaluation')
                valid_loss, valid_acc, valid_bleu = eval_epoch(
                    model, validation_data, device, opt.beam_size, opt.max_token_seq_len
                )

                # save
                if valid_bleu > max_bleu or opt.save_mode == 'all':
                    if opt.save_mode == 'all':
                        path = opt.save_model + f'_accuracy_{round(100*valid_acc, 3)}'
                    path = opt.save_model + '.chkpt'
                    checkpoint = {
                        'model': model.state_dict(),
                        'settings': opt,
                        'epoch': epoch_i,
                        'step': global_step
                    }
                    torch.save(checkpoint, path)
                    print(f'    - [Info] The checkpoint file has been saved as {path}.')
            # end of evaluate

            # prepare data
            # pos - position, it is added in collate_fn
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            optimizer.zero_grad()
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

            # backward
            loss, n_correct = cal_performance(pred, gold, smoothing=opt.label_smoothing)
            loss.backward()

            # update parameters
            optimizer.step_and_update_lr()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            wandb.log({
                'training_loss': loss.item() / n_word,
                'accuracy': n_correct / n_word
            })

        train_loss = loss.item()
        train_acc = n_correct / n_word
        print(f'  - (Training)   ppl: {round(perplexity(train_loss), 5)}, '
              f'accuracy: {round(train_acc, 3)} %, '
              f'elapse: {round((time.time()-start)/60, 3)} min')

    # end for
    # TODO: load best model and test it


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True,
                        help='path to dataset.pt. If -wmt flag is specified, path to config.json')

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-num_workers', type=int, default=4)

    # parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_factor', type=float, default=1.0)
    parser.add_argument('-eval_every', type=int, default=10_000,
                        help='validate (and save model) every n batches')
    parser.add_argument('-beam_size', type=int, default=3,
                        help='beam size for validation BLEU score')

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-unet', action='store_true')

    parser.add_argument('-single_gpu', action='store_true')

    # for torch.distributed.launch
    parser.add_argument('--local_rank', type=int, help='GPU (group) number to use')

    opt = parser.parse_args()

    if not opt.single_gpu:
        distr.init_process_group(backend='nccl')

    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # ========= Loading Dataset ========= #
    with open(opt.data) as f:
        data_config = json.load(f)

    src_bpe = yttm.BPE(model=data_config['src_bpe'])
    tgt_bpe = yttm.BPE(model=data_config['tgt_bpe'])

    training_dataset = RAFTranslationDataset(
        data_config['train_src'],
        data_config['train_tgt'],
        lambda x: src_bpe.encode([x], output_type=yttm.OutputType.ID, bos=True, eos=True)[0],
        lambda x: tgt_bpe.encode([x], output_type=yttm.OutputType.ID, bos=True, eos=True)[0],
        max_len=data_config['max_word_seq_len'],
        warm_up=True
    )

    validation_dataset = RAFTranslationDataset(
        data_config['valid_src'],
        data_config['valid_tgt'],
        lambda x: src_bpe.encode([x], output_type=yttm.OutputType.ID, bos=True, eos=True)[0],
        lambda x: tgt_bpe.encode([x], output_type=yttm.OutputType.ID, bos=True, eos=True)[0],
        max_len=data_config['max_word_seq_len'],
        warm_up=True,
        tgt_detokenizer=lambda x: tgt_bpe.decode(x)[0]
    )

    training_sampler = RandomSampler(training_dataset)
    validation_sampler = RandomSampler(validation_dataset)

    if not opt.single_gpu:
        training_sampler = DistributedSampler(training_dataset)
        validation_sampler = DistributedSampler(validation_dataset)

    training_data = torch.utils.data.DataLoader(
        training_dataset,
        num_workers=opt.num_workers, batch_size=opt.batch_size, collate_fn=paired_collate_fn,
        sampler=training_sampler
    )

    validation_data = torch.utils.data.DataLoader(
        validation_dataset,
        num_workers=opt.num_workers, batch_size=opt.batch_size, collate_fn=paired_collate_fn,
        sampler=validation_sampler
    )

    # additional opt fields for coppatibility with original training script
    opt.src_vocab_size = src_bpe.vocab_size()
    opt.tgt_vocab_size = tgt_bpe.vocab_size()
    opt.max_token_seq_len = data_config['max_word_seq_len']

    # ========= Preparing Model ========= #

    print(opt)

    device_str = 'cuda' if opt.cuda else 'cpu'
    device = torch.device(device_str, opt.local_rank)

    model = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        unet=opt.unet).to(device)

    if not opt.single_gpu:
        print(f'[Info] using {torch.cuda.device_count()} GPUs (in distributed mode)')
        model = WrappedDistributedDataParallel(
            model, device_ids=[opt.local_rank], output_device=opt.local_rank
        )

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps, lr_factor=opt.lr_factor)

    wandb.init(project='hierarchical_transformer', config=opt, notes='main run')
    wandb.watch(model)

    os.makedirs(os.path.dirname(opt.save_model), exist_ok=True)

    train(model, training_data, validation_data, optimizer, device, opt)


if __name__ == '__main__':
    main()
