# coding: utf-8
# author: ztypl
# date:   2021/9/1

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import dgl
from copy import deepcopy

"""
referenced codes in https://github.com/dmlc/dgl/blob/master/examples/pytorch/tree_lstm/tree_lstm.py
https://arxiv.org/abs/1503.00075
"""




class Tree:
    def __init__(self, h_size, x_size):
        self.dgl_graph = dgl.DGLGraph()
        self.h_size = h_size
        self.x_size = x_size

    def add_node(self, token:int, topic, parent_id=None, tensor:torch.Tensor = torch.Tensor()):
        self.dgl_graph.add_nodes(1, data={'t': torch.LongTensor([token]),
                                          'topic': torch.FloatTensor(topic).unsqueeze(0),
                                          'x': tensor.new_zeros(size=(1, self.x_size)),
                                          'h': tensor.new_zeros(size=(1, self.h_size)),
                                          'c': tensor.new_zeros(size=(1, self.h_size))})
        added_node_id = self.dgl_graph.number_of_nodes() - 1
        if parent_id:
            self.dgl_graph.add_edge(added_node_id, parent_id)
        return added_node_id

    def add_node_bottom_up(self, token:int, topic, child_ids, tensor: torch.Tensor = torch.Tensor()):
        self.dgl_graph.add_nodes(1, data={'t': torch.LongTensor([token]),
                                          'topic': torch.FloatTensor(topic).unsqueeze(0),
                                          'x': tensor.new_zeros(size=(1, self.x_size)),
                                          'h': tensor.new_zeros(size=(1, self.h_size)),
                                          'c': tensor.new_zeros(size=(1, self.h_size))})
        added_node_id = self.dgl_graph.number_of_nodes() - 1
        for child_id in child_ids:
            self.dgl_graph.add_edge(child_id, added_node_id)
        return added_node_id

    def add_link(self, child_id, parent_id):
        self.dgl_graph.add_edge(child_id, parent_id)


class BatchedTree:
    def __init__(self, tree_list):
        graph_list = []
        for tree in tree_list:
            graph_list.append(tree.dgl_graph)
        self.batch_dgl_graph = dgl.batch(graph_list)

    def get_hidden_state(self):
        graph_list = dgl.unbatch(self.batch_dgl_graph)
        hidden_states = []
        max_nodes_num = max([len(graph.nodes()) for graph in graph_list])
        for graph in graph_list:
            hiddens = graph.ndata['h']
            node_num, hidden_num = hiddens.size()
            if len(hiddens) < max_nodes_num:
                padding = hiddens.new_zeros(size=(max_nodes_num - node_num, hidden_num))
                hiddens = torch.cat((hiddens, padding), dim=0)
            hidden_states.append(hiddens)
        return torch.stack(hidden_states)

    def cuda(self):
        self.batch_dgl_graph = self.batch_dgl_graph.to(f'cuda:{torch.cuda.current_device()}')
        return self


class TreeLSTM(torch.nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 topic_size,
                 num_vocabs,
                 dropout,
                 cell_type='n_ary',
                 n_ary=None,
                 num_stacks=2):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.topic_size = topic_size
        self.embedding = nn.Embedding(num_vocabs, x_size)
        self.dropout = torch.nn.Dropout(dropout)
        # if cell_type == 'n_ary':
        #     self.cell = NaryTreeLSTMCell(n_ary, x_size, h_size)
        # else:
        #     self.cell = ChildSumTreeLSTMCell(x_size, h_size)
        self.cell_type = NaryTreeLSTMCell if cell_type == 'n_ary' else ChildSumTreeLSTMCell
        self.num_stacks = num_stacks
        self.cells = nn.ModuleList([self.cell_type(n_ary, x_size+topic_size, h_size)])
        for i in range(1, num_stacks):
            self.cells.append(self.cell_type(n_ary, h_size, h_size))

    def forward(self, batch: BatchedTree):
        batches = [deepcopy(batch) for _ in range(self.num_stacks)]
        for stack in range(self.num_stacks):
            cur_batch = batches[stack]
            if stack > 0:
                prev_batch = batches[stack - 1]
                cur_batch.batch_dgl_graph.ndata['x'] = prev_batch.batch_dgl_graph.ndata['h']
            else:
                cur_batch.batch_dgl_graph.ndata['x'] = torch.cat([
                    self.embedding(cur_batch.batch_dgl_graph.ndata['t']),
                    cur_batch.batch_dgl_graph.ndata['topic']
                ], dim=1)
            # cur_batch.batch_dgl_graph.update_all(self.cell.message_func)
            # cur_batch.batch_dgl_graph.register_reduce_func(self.cell.reduce_func)
            # cur_batch.batch_dgl_graph.register_apply_node_func(self.cell.apply_node_func)
            cur_batch.batch_dgl_graph.ndata['iou'] = self.cells[stack].W_iou(self.dropout(cur_batch.batch_dgl_graph.ndata['x']))

            dgl.prop_nodes_topo(cur_batch.batch_dgl_graph, self.cells[stack].message_func, self.cells[stack].reduce_func,
                                apply_node_func=self.cells[stack].apply_node_func)

        output = pad_sequence(batches[-1].batch_dgl_graph.ndata['h'].split(batches[-1].batch_dgl_graph.batch_num_nodes().tolist()))
        hidden = torch.stack([batch.batch_dgl_graph.ndata['h'][batch.batch_dgl_graph.batch_num_nodes().cumsum(axis=0) - 1] for batch in batches])
        return output, hidden


class NaryTreeLSTMCell(torch.nn.Module):
    def __init__(self, n_ary, x_size, h_size):
        super(NaryTreeLSTMCell, self).__init__()
        self.n_ary = n_ary
        self.h_size = h_size
        self.W_iou = torch.nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = torch.nn.Linear(n_ary * h_size, 3 * h_size, bias=False)
        self.b_iou = torch.nn.Parameter(torch.zeros(1, 3 * h_size), requires_grad=True)
        self.U_f = torch.nn.Linear(n_ary * h_size, n_ary * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        padding_hs = self.n_ary - nodes.mailbox['h'].size(1)
        padding = h_cat.new_zeros(size=(nodes.mailbox['h'].size(0), padding_hs * self.h_size))
        h_cat = torch.cat((h_cat, padding), dim=1)
        f = torch.sigmoid(self.U_f(h_cat)).view(nodes.mailbox['h'].size(0), self.n_ary, self.h_size)
        padding_cs = self.n_ary - nodes.mailbox['c'].size(1)
        padding = h_cat.new_zeros(size=(nodes.mailbox['c'].size(0), padding_cs, self.h_size))
        c = torch.cat((nodes.mailbox['c'], padding), dim=1)
        c = torch.sum(f * c, 1)
        return {'iou': nodes.data['iou'] + self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}


class ChildSumTreeLSTMCell(torch.nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = torch.nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = torch.nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = torch.nn.Parameter(torch.zeros(1, 3 * h_size), requires_grad=True)
        self.U_f = torch.nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox['h']))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iou': nodes.data['iou'] + self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.em_dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Choose attention model
        self.attn = Attn(hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.em_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_size)  # S=1 x B x N

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        output = self.out(torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

        # Return final output, hidden state
        return output, hidden