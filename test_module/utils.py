import json
import torch
import numpy as np
import h5py
import torch
from model import ModuleNet, Seq2Seq

SPECIAL_TOKENS = {
  '<NULL>': 0,
  '<START>': 1,
  '<END>': 2,
  '<UNK>': 3,
}

def tokenize(s, delim=' ',
      add_start_token=True, add_end_token=True,
      punct_to_keep=None, punct_to_remove=None):
  if punct_to_keep is not None:
    for p in punct_to_keep:
      s = s.replace(p, '%s%s' % (delim, p))

  if punct_to_remove is not None:
    for p in punct_to_remove:
      s = s.replace(p, '')

  tokens = s.split(delim)
  if add_start_token:
    tokens.insert(0, '<START>')
  if add_end_token:
    tokens.append('<END>')
  return tokens

def encode(seq_tokens, token_to_idx, allow_unk=False):
  seq_idx = []
  for token in seq_tokens:
    if token not in token_to_idx:
      if allow_unk:
        token = '<UNK>'
      else:
        raise KeyError('Token "%s" not in vocab' % token)
    seq_idx.append(token_to_idx[token])
  return seq_idx

def function_to_str(f):
  value_str = ''
  if f['value_inputs']:
    value_str = '[%s]' % ','.join(f['value_inputs'])
  return '%s%s' % (f['function'], value_str)


def str_to_function(s):
  if '[' not in s:
    return {
      'function': s,
      'value_inputs': [],
    }
  name, value_str = s.replace(']', '').split('[')
  return {
    'function': name,
    'value_inputs': value_str.split(','),
  }


def list_to_str(program_list):
  return ' '.join(function_to_str(f) for f in program_list)


def get_num_inputs(f):
  if type(f) is str:
    f = str_to_function(f)
  name = f['function']
  if name == 'scene':
    return 0
  if 'equal' in name or name in ['union', 'intersect', 'less_than', 'greater_than']:
    return 2
  return 1

def load_program_generator(path):
  checkpoint = torch.load(path, map_location=lambda storage, loc: storage)#load_cpu(path)
  kwargs = checkpoint['program_generator_kwargs']
  state = checkpoint['program_generator_state']
  model = Seq2Seq(**kwargs)
  model.load_state_dict(state)
  return model, kwargs


def load_execution_engine(path, verbose=True):
  checkpoint = torch.load(path, map_location=lambda storage, loc: storage) #load_cpu(path)
  kwargs = checkpoint['execution_engine_kwargs']
  state = checkpoint['execution_engine_state']
  kwargs['verbose'] = verbose
  model = ModuleNet(**kwargs)
  cur_state = model.state_dict()
  model.load_state_dict(state)
  return model, kwargs