import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    '''
    GAT(nhid, nhid, nhid, 0.5, 0.2, 8)
    '''
    def __init__(self, nfeat, nhid, nclass, dropout=0.0, alpha=0.2, nheads=1):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        input_x = x
        x = F.dropout(x, self.dropout)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout)
        x = F.elu(self.out_att(x, adj))
        # residual connection
        return x + input_x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        '''
        input: [B, L, H_in]
        adj: [B, L, L]
        '''
        h = self.W(input)  # [B, L, H_out]
        L = h.size()[1]  # L
        a_input = torch.cat([h.repeat(1, 1, L).view(h.shape[0], L * L, -1), h.repeat(1, L, 1)], dim=2)\
            .view(h.shape[0], L, -1, 2 * self.out_features)  # [B, L, L, 2*H_out]
        e = self.leakyrelu(self.a(a_input).squeeze(3))  # [B, L, L]

        zero_vec = -99999999 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = []

        for per_a, per_h in zip(attention, h):
            h_prime.append(torch.matmul(per_a, per_h))

        h_prime = torch.stack(h_prime)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime