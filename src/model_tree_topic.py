# coding: utf-8
# author: ztypl
# date:   2021/11/11


import torch
import torch.nn as nn
from tree_module import *


class EncoderTreeLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, op_set, n_layers=2, dropout=0.5):
        super(EncoderTreeLSTM, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.op_set = op_set

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.treelstm = TreeLSTM(input_size, hidden_size, dropout, cell_type='n_ary', num_stacks=n_layers)
        # self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)

        tree_list = []
        for input_seq in input_seqs:
            stack = []


        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)


        # packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        # outputs, hidden = self.gru(packed, hidden)
        # outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        # S x B x H
        return outputs, hidden