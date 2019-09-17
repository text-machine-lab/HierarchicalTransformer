'''
This script handling the training process.
'''

import argparse
import math
import time
import json

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from translation_dataset import TranslationDataset, RAFTranslationDataset, paired_collate_fn
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from torch.utils.tensorboard import SummaryWriter
import youtokentome as yttm
import datetime

import wandb

writer = None


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['train']['src'],
            tgt_insts=data['train']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

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


def train_epoch(epoch_idx, model, training_data, optimizer, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch_idx, batch in enumerate(tqdm(training_data, desc='  - (Training)   ', leave=False)):

        # prepare data
        # pos - position, it is added in collate_fn
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

        wandb.log({
            'training_loss': loss.item() / n_word, 
        })

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch(epoch_idx, model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data,
                desc='  - (Validation) ', leave=False):

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

    loss_per_word = total_loss/n_word_total

    wandb.log({'val_loss': loss_per_word})
    writer.add_scalar('val_loss', loss_per_word, epoch_idx)

    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(epoch_i,
            model, training_data, optimizer, device, smoothing=opt.label_smoothing)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(epoch_i, model, validation_data, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

    # This code allows for saving a randomly initialized model
    if opt.epoch == 0:
        model_state_dict = model.state_dict()
        model_name = opt.save_model + '.chkpt'
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': 0}
        torch.save(checkpoint, model_name)
        print('    - [Info] Zero epochs run, so model is random. The checkpoint file has been updated.')


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True,
                        help='path to dataset.pt. If -wmt flag is specified, path to config.json')

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)

    #parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_factor', type=float, default=1.0)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-unet', action='store_true')

    parser.add_argument('-wmt', action='store_true',
                        help=('preprocess data on the fly machine translation dataloader '
                              'consumes much less memory'))  # How -data is used?
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # new code to run Tensorboard
    global writer
    writer = SummaryWriter('logdir/' + str(datetime.datetime.now()) + ' ' + str(opt))

    #========= Loading Dataset =========#
    if not opt.wmt:
        data = torch.load(opt.data)
        opt.max_token_seq_len = data['settings'].max_token_seq_len

        training_data, validation_data = prepare_dataloaders(data, opt)

        opt.src_vocab_size = training_data.dataset.src_vocab_size
        opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    else:
        with open(opt.data) as f:
            data_config = json.load(f)

        src_bpe = yttm.BPE(model=data_config['src_bpe'])
        tgt_bpe = yttm.BPE(model=data_config['tgt_bpe'])

        training_dataset = RAFTranslationDataset(
            data_config['train_src'],
            data_config['train_tgt'],
            lambda x: src_bpe.encode([x], output_type=yttm.OutputType.ID, bos=True, eos=True)[0],
            lambda x: tgt_bpe.encode([x], output_type=yttm.OutputType.ID, bos=True, eos=True)[0],
            max_len=data_config['max_word_seq_len']
        )

        validation_dataset = RAFTranslationDataset(
            data_config['valid_src'],
            data_config['valid_tgt'],
            lambda x: src_bpe.encode([x], output_type=yttm.OutputType.ID, bos=True, eos=True)[0],
            lambda x: tgt_bpe.encode([x], output_type=yttm.OutputType.ID, bos=True, eos=True)[0],
            max_len=data_config['max_word_seq_len']
        )

        # TODO: fix multiple workers
        training_data = torch.utils.data.DataLoader(training_dataset,
            num_workers=2, batch_size=opt.batch_size, collate_fn=paired_collate_fn, shuffle=True)

        validation_data = torch.utils.data.DataLoader(validation_dataset,
            num_workers=2, batch_size=opt.batch_size, collate_fn=paired_collate_fn, shuffle=False)

        opt.src_vocab_size = src_bpe.vocab_size()
        opt.tgt_vocab_size = tgt_bpe.vocab_size()
        opt.max_token_seq_len = data_config['max_word_seq_len']

    #========= Preparing Model =========#
    if opt.embs_share_weight and not opt.wmt:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    device = torch.device('cuda' if opt.cuda else 'cpu')
    transformer = Transformer(
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

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps, lr_factor=opt.lr_factor)

    wandb.init(project='hierarchical_transformer', config=opt, notes='Debug run on small dataset')
    wandb.watch(transformer)

    train(transformer, training_data, validation_data, optimizer, device, opt)


if __name__ == '__main__':

    main()

    # must close writer to save changes to Tensorboard
    writer.close()
