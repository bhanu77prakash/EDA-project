import argparse
import json
import random
import shutil
import sys
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import numpy as np
import h5py
from scipy.misc import imread, imresize

from utils import *
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--program_generator', default=None)
parser.add_argument('--execution_engine', default=None)
parser.add_argument('--question', default=None)
parser.add_argument('--image', default=None)


def main(args):
  print()
  model = None
  print('Loading program generator ')
  program_generator, _ = load_program_generator('models/program_generator.pt')
  print('Loading execution engine ')
  execution_engine, _ = load_execution_engine('models/execution_engine.pt', verbose=False)
  model = (program_generator, execution_engine)
  (ans,lis)=run_single_example(args, model)
  with open('ans.pkl', 'wb') as f:
  	pickle.dump(ans, f)
  with open('lis.pkl', 'wb') as f:
  	pickle.dump(lis, f)

def run_single_example(args, model):
  dtype = torch.FloatTensor
  print('Loading CNN for feature extraction')
  cnn = build_cnn(args, dtype)

  # Load and preprocess the image
  img_size = (224,224)
  img = imread(args.image, mode='RGB')
  img = imresize(img, img_size, interp='bicubic')
  img = img.transpose(2, 0, 1)[None]
  mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
  std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
  img = (img.astype(np.float32) / 255.0 - mean) / std

  # Use CNN to extract features for the image
  img_var = Variable(torch.FloatTensor(img).type(dtype), volatile=True)
  feats_var = cnn(img_var)

  # Tokenize the question
  vocab =torch.load('models/program_generator.pt', map_location=lambda storage, loc: storage)['vocab']#load_vocab(args)
  question_tokens = tokenize(args.question,punct_to_keep=[';', ','],punct_to_remove=['?', '.'])
  question_encoded = encode(question_tokens,vocab['question_token_to_idx'],allow_unk=True)
  question_encoded = torch.LongTensor(question_encoded).view(1, -1)
  question_encoded = question_encoded.type(dtype).long()
  question_var = Variable(question_encoded, volatile=True)

  # Run the model
  print('Running the model\n')
  scores = None
  predicted_program = None
  if type(model) is tuple:
    program_generator, execution_engine = model
    program_generator.type(dtype)
    execution_engine.type(dtype)
    predicted_program = program_generator.reinforce_sample(question_var,temperature=1.0,argmax=True)
    scores = execution_engine(feats_var, predicted_program)

  # Print results
  _, predicted_answer_idx = scores.data.cpu()[0].max(dim=0)
  predicted_answer = vocab['answer_idx_to_token'][predicted_answer_idx[0]]

  print('Question: "%s"' % args.question)
  print('Predicted answer: ', predicted_answer)
  ans1=predicted_answer
  lis1=[]
  if predicted_program is not None:
    print()
    print('Predicted program:')
    program = predicted_program.data.cpu()[0]
    num_inputs = 1
    for fn_idx in program:
      fn_str = vocab['program_idx_to_token'][fn_idx]
      num_inputs += get_num_inputs(fn_str) - 1
      print(fn_str)
      lis1.append(fn_str)
      if num_inputs == 0:
        break
  return(ans1,lis1)


def build_cnn(args, dtype):
  whole_cnn = getattr(torchvision.models, 'resnet101')(pretrained=True)
  layers = [whole_cnn.conv1,whole_cnn.bn1,whole_cnn.relu,whole_cnn.maxpool,]
  for i in range(3):
    name = 'layer%d' % (i + 1)
    layers.append(getattr(whole_cnn, name))
  cnn = torch.nn.Sequential(*layers)
  cnn.type(dtype)
  cnn.eval()
  return cnn

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
