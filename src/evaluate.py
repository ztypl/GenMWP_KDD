# coding: utf-8
# author: ztypl
# date:   2021/4/26

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence

from .model import *
from .preprocess import *
from .masked_cross_entropy import *

def eval_attn(input_seq,
               encoder: EncoderRNN, decoder: AttnDecoderRNN,
               output_lang,
               use_cuda=True):
    encoder.eval()
    decoder.eval()

    input_seq = torch.LongTensor(input_seq).unsqueeze(0)
    input_len = input_seq.size()
    seq_mask = torch.zeros((1, input_len))

    if use_cuda:
        input_seq = input_seq.cuda()
        seq_mask = seq_mask.cuda()


