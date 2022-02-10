# coding: utf-8
# author: ztypl
# date:   2021/4/22

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence

from .model import *
from .preprocess import *
from .masked_cross_entropy import *


def train_attn(input_batch, target_batch,
               encoder, decoder,
               encoder_optimizer, decoder_optimizer,
               output_lang,
               use_cuda=True):
    encoder.train()
    decoder.train()

    if use_cuda:
        input_batch = input_batch.cuda()
        target_batch = target_batch.cuda()

    input_seqs, input_lens = pad_packed_sequence(input_batch, batch_first=True)  # B * S, B
    target_seqs, target_lens = pad_packed_sequence(target_batch, batch_first=True)  # B * S, B
    max_len = input_lens.max()
    batch_size = input_lens.size(0)
    seq_mask = (torch.arange(max_len).expand(len(input_lens), max_len) >= input_lens.unsqueeze(1))
    if use_cuda:
        seq_mask = seq_mask.cuda()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_output, encoder_hidden = encoder(input_seqs.T, input_lens, None)
    decoder_input = torch.LongTensor([output_lang.word2index[Tokens.SOS]] * batch_size)
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    max_target_length = max(target_lens)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    if use_cuda:
        all_decoder_outputs = all_decoder_outputs.cuda()

    # teacher forcing
    for t in range(max_target_length):
        if use_cuda:
            decoder_input = decoder_input.cuda()
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output, seq_mask)
        all_decoder_outputs[t] = decoder_output
        decoder_input = target_seqs[:, t]

    loss = masked_cross_entropy(all_decoder_outputs.transpose(0, 1).contiguous(), target_seqs.contiguous(), target_lens)
    loss.backward()
    return_loss = loss.item()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return return_loss
