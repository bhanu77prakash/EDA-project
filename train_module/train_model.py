import sys
import os

import argparse
import json
import random
import shutil

import torch
torch.backends.cudnn.enabled = True
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import h5py

from utils import *
from utils import ClevrDataset, ClevrDataLoader
from model import ModuleNet, Seq2Seq


parser = argparse.ArgumentParser()
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--model_type', default='PG',choices=['PG', 'EE', 'PG+EE'])
parser.add_argument('--num_iterations', default=100000, type=int)
parser.add_argument('--checkpoint_path', default='data/checkpoint.pt')
parser.add_argument('--checkpoint_every', default=10000, type=int)


def main(args):
  vocab = load_vocab('data/vocab.json')
  question_families = None

  train_loader_kwargs = {
    'question_h5': 'data/train_questions.h5',
    'feature_h5': 'data/train_features.h5',
    'vocab': vocab,
    'batch_size': 64,
    'shuffle': True,
    'question_families': question_families,
    'max_samples': args.num_train_samples,
    'num_workers': 1,
  }
  val_loader_kwargs = {
    'question_h5': 'data/val_questions.h5',
    'feature_h5': 'data/val_features.h5',
    'vocab': vocab,
    'batch_size': 64,
    'question_families': question_families,
    'max_samples': 10000,
    'num_workers': 1,
  }

  with ClevrDataLoader(**train_loader_kwargs) as train_loader, \
       ClevrDataLoader(**val_loader_kwargs) as val_loader:
    train_loop(args, train_loader, val_loader)


def train_loop(args, train_loader, val_loader):
  vocab = load_vocab('data/vocab.json')
  program_generator, pg_kwargs, pg_optimizer = None, None, None
  execution_engine, ee_kwargs, ee_optimizer = None, None, None

  pg_best_state, ee_best_state = None, None

  # Set up model
  if args.model_type == 'PG' or args.model_type == 'PG+EE':
    program_generator, pg_kwargs = get_program_generator(args)
    pg_optimizer = torch.optim.Adam(program_generator.parameters(),lr=5e-4)
    print('Here is the program generator:')
    print(program_generator)

  if args.model_type == 'EE' or args.model_type == 'PG+EE':
    execution_engine, ee_kwargs = get_execution_engine(args)
    ee_optimizer = torch.optim.Adam(execution_engine.parameters(),lr=5e-4)
    print('Here is the execution engine:')
    print(execution_engine)

  loss_fn = torch.nn.CrossEntropyLoss().cuda()

  stats = {
    'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
    'train_accs': [], 'val_accs': [], 'val_accs_ts': [],
    'best_val_acc': -1, 'model_t': 0,
  }
  t, epoch, reward_moving_average = 0, 0, 0

  set_mode('train', [program_generator, execution_engine])

  print('train_loader has %d samples' % len(train_loader.dataset))
  print('val_loader has %d samples' % len(val_loader.dataset))

  while t < args.num_iterations:
    epoch += 1
    print('Starting epoch %d' % epoch)
    for batch in train_loader:
      t += 1
      questions, _, feats, answers, programs, _ = batch
      questions_var = Variable(questions.cuda())
      feats_var = Variable(feats.cuda())
      answers_var = Variable(answers.cuda())
      if programs[0] is not None:
        programs_var = Variable(programs.cuda())

      reward = None
      if args.model_type == 'PG':
        # Train program generator with ground-truth programs
        pg_optimizer.zero_grad()
        loss = program_generator(questions_var, programs_var)
        loss.backward()
        pg_optimizer.step()
      elif args.model_type == 'EE':
        # Train execution engine with ground-truth programs
        ee_optimizer.zero_grad()
        scores = execution_engine(feats_var, programs_var)
        loss = loss_fn(scores, answers_var)
        loss.backward()
        ee_optimizer.step()
      elif args.model_type == 'PG+EE':
        programs_pred = program_generator.reinforce_sample(questions_var)
        scores = execution_engine(feats_var, programs_pred)

        loss = loss_fn(scores, answers_var)
        _, preds = scores.data.cpu().max(1)
        raw_reward = (preds == answers).float()
        reward_moving_average *= 0.9
        reward_moving_average += (1.0 - 0.9) * raw_reward.mean()
        centered_reward = raw_reward - reward_moving_average

		ee_optimizer.zero_grad()
		loss.backward()
		ee_optimizer.step()

		pg_optimizer.zero_grad()
		program_generator.reinforce_backward(centered_reward.cuda())
		pg_optimizer.step()

      
      if t % args.checkpoint_every == 0:
        print('Checking training accuracy ... ')
        train_acc = check_accuracy(args, program_generator, execution_engine,train_loader)
        print('train accuracy is', train_acc)
        print('Checking validation accuracy ...')
        val_acc = check_accuracy(args, program_generator, execution_engine,val_loader)
        print('val accuracy is ', val_acc)
        stats['train_accs'].append(train_acc)
        stats['val_accs'].append(val_acc)
        stats['val_accs_ts'].append(t)

        if val_acc > stats['best_val_acc']:
          stats['best_val_acc'] = val_acc
          stats['model_t'] = t
          best_pg_state = get_state(program_generator)
          best_ee_state = get_state(execution_engine)

        checkpoint = {
          'args': args.__dict__,
          'program_generator_kwargs': pg_kwargs,
          'program_generator_state': best_pg_state,
          'execution_engine_kwargs': ee_kwargs,
          'execution_engine_state': best_ee_state,
          'vocab': vocab
        }
        for k, v in stats.items():
          checkpoint[k] = v
        print('Saving checkpoint to %s' % args.checkpoint_path)
        torch.save(checkpoint, args.checkpoint_path)
        del checkpoint['program_generator_state']
        del checkpoint['execution_engine_state']
        with open(args.checkpoint_path + '.json', 'w') as f:
          json.dump(checkpoint, f)

      if t == args.num_iterations:
        break


def parse_int_list(s):
  return tuple(int(n) for n in s.split(','))


def get_state(m):
  if m is None:
    return None
  state = {}
  for k, v in m.state_dict().items():
    state[k] = v.clone()
  return state


def get_program_generator(args):
  vocab = load_vocab('data/vocab.json')
  kwargs = {
    'encoder_vocab_size': len(vocab['question_token_to_idx']),
    'decoder_vocab_size': len(vocab['program_token_to_idx']),
    'wordvec_dim': 300,
    'hidden_dim': 256,
    'rnn_num_layers': 2,
    'rnn_dropout': 0,
  }
  pg = Seq2Seq(**kwargs)
  pg.cuda()
  pg.train()
  return pg, kwargs


def get_execution_engine(args):
  vocab = load_vocab('data/vocab.json')
  kwargs = {
    'vocab': vocab,
    'feature_dim': parse_int_list(args.feature_dim),
    'stem_batchnorm': False,
    'stem_num_layers': 2,
    'module_dim': 128,
    'module_residual': True,
    'module_batchnorm': False,
    'classifier_proj_dim': 512,
    'classifier_downsample': 'maxpool2',
    'classifier_fc_layers': parse_int_list(0),
    'classifier_batchnorm': False,
    'classifier_dropout': 0,
  }
  ee = ModuleNet(**kwargs)
  ee.cuda()
  ee.train()
  return ee, kwargs

def set_mode(mode, models):
  assert mode in ['train', 'eval']
  for m in models:
    if m is None: continue
    if mode == 'train': m.train()
    if mode == 'eval': m.eval()

def check_accuracy(args, program_generator, execution_engine, loader):
  set_mode('eval', [program_generator, execution_engine])
  num_correct, num_samples = 0, 0
  for batch in loader:
    questions, _, feats, answers, programs, _ = batch

    questions_var = Variable(questions.cuda(), volatile=True)
    feats_var = Variable(feats.cuda(), volatile=True)
    answers_var = Variable(feats.cuda(), volatile=True)
    if programs[0] is not None:
      programs_var = Variable(programs.cuda(), volatile=True)

    scores = None # Use this for everything but PG
    if args.model_type == 'PG':
      vocab = load_vocab('data/vocab.json')
      for i in range(questions.size(0)):
        program_pred = program_generator.sample(Variable(questions[i:i+1].cuda(), volatile=True))
        program_pred_str = iep.preprocess.decode(program_pred, vocab['program_idx_to_token'])
        program_str = iep.preprocess.decode(programs[i], vocab['program_idx_to_token'])
        if program_pred_str == program_str:
          num_correct += 1
        num_samples += 1
    elif args.model_type == 'EE':
        scores = execution_engine(feats_var, programs_var)
    elif args.model_type == 'PG+EE':
      programs_pred = program_generator.reinforce_sample(
                          questions_var, argmax=True)
      scores = execution_engine(feats_var, programs_pred)
      
    if scores is not None:
      _, preds = scores.data.cpu().max(1)
      num_correct += (preds == answers).sum()
      num_samples += preds.size(0)

    if num_samples >= 10000:
      break

  set_mode('train', [program_generator, execution_engine])
  acc = float(num_correct) / num_samples
  return acc


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
