import json
import torch
import numpy as np
import h5py
import torch
from model import ModuleNet, Seq2Seq

def invert_dict(d):
  return {v: k for k, v in d.items()}


def load_vocab(path):
  with open(path, 'r') as f:
    vocab = json.load(f)
    vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
    vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
  assert vocab['question_token_to_idx']['<NULL>'] == 0
  assert vocab['question_token_to_idx']['<START>'] == 1
  assert vocab['question_token_to_idx']['<END>'] == 2
  assert vocab['program_token_to_idx']['<NULL>'] == 0
  assert vocab['program_token_to_idx']['<START>'] == 1
  assert vocab['program_token_to_idx']['<END>'] == 2
  return vocab


def load_program_generator(path):
  checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
  kwargs = checkpoint['program_generator_kwargs']
  state = checkpoint['program_generator_state']
  model = Seq2Seq(**kwargs)
  model.load_state_dict(state)
  return model, kwargs


def load_execution_engine(path, verbose=True):
  checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
  kwargs = checkpoint['execution_engine_kwargs']
  state = checkpoint['execution_engine_state']
  kwargs['verbose'] = verbose
  model = ModuleNet(**kwargs)
  cur_state = model.state_dict()
  model.load_state_dict(state)
  return model, kwargs
