from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
from collections import Counter


# torch.cuda.set_device(1)
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SelfAttentionWide(nn.Module):

    def __init__(self, emb, heads=8, mask=False):
        super().__init__()

        self.emb = emb
        self.heads = heads
        # self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):
        b = 1
        t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dimension {{e}} should match layer embedding dim {{self.emb}}'

        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # dot-product attention

        # folding heads to batch dimensions

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)

        # if self.mask:
        #     mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, e)

        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)



#   reconstruct graph
def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z, dec_h1, dec_h2, dec_h3  # 将encoder和decoder都返回


class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # # GCN for inter information
        # self.gnn_1 = GNNLayer(n_input, n_enc_1)
        # self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        # self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        # self.gnn_4 = GNNLayer(n_enc_3, n_z)
        # self.gnn_5 = GNNLayer(n_z, n_clusters)

        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        # self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)#
        self.gnn_3 = GNNLayer(n_enc_1, n_z)  #
        # self.gnn_4 = GNNLayer(n_enc_3, n_z)#
        self.gnn_5 = GNNLayer(n_z, n_clusters)
        self.gnn_6 = GNNLayer(n_clusters, n_dec_1)
        self.gnn_7 = GNNLayer(n_dec_1, n_dec_2)
        # self.gnn_8 = GNNLayer(n_dec_2, n_dec_3)
        self.gnn_9 = GNNLayer(n_dec_2, n_input)
        self.attn1 = SelfAttentionWide(n_enc_1)
        self.attn2 = SelfAttentionWide(n_enc_2)
        self.attn3 = SelfAttentionWide(n_enc_1)
        self.attn4 = SelfAttentionWide(n_z)
        self.attn5 = SelfAttentionWide(n_z)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z, dec_1, dec_2, dec_3 = self.ae(x)  # 增加了decoder的输出
        # print(x.shape)

        # GCN Module
        h = self.gnn_1(x, adj)  # 相加

        h = self.attn3((h + tra2))
        h = h.squeeze(0)
        h = self.gnn_3(h, adj)

        # print(h.shape)
        h1 = self.attn5((h+z))
        h1 = h1.squeeze(0)
        h1 = self.gnn_5(h1, adj, active=False)
        # h1.unsqueeze(0)
        # print(h1.shape)
        # h1 = self.attn5(h1)
        # h1 = h1.squeeze(0)
        # print(h1.shape)
        predict = F.softmax(h1, dim=1)
        h = self.gnn_6(h1, adj)
        h = self.gnn_7(h + dec_1, adj)
        # h = self.gnn_8(h + dec_2, adj)
        h = self.gnn_9(h + dec_3, adj)
        A_pred = dot_product_decode(h)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, h , A_pred ,h1


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(dataset):
    model = SDCN(500, 500, 2000, 2000, 500, 500,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.name, args.k)
    # adj = adj
    # adj = adj
    adj = adj.to(device)
    # pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    # weight_mask = adj.to_dense().view(-1) == 1
    # weight_tensor = torch.ones(weight_mask.size(0))
    # weight_tensor[weight_mask] = pos_weight
    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z, _, _, _ = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, max_iter=1000)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')

    for epoch in range(700):
        if epoch % 1 == 0:
            # update_interval
            _, tmp_q, pred, _, h , A_pred, h1 = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)  # 计算p是一种自增强

            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            # print(type(res1))
            res2 = pred.data.cpu().numpy().argmax(1)  # Z
            # print(type(res2))
            res3 = p.data.cpu().numpy().argmax(1)  # P
            eva(y, res1, str(epoch) + 'Q')
            eva(y, res2, str(epoch) + 'Z')
            eva(y, res3, str(epoch) + 'P')

        x_bar, q, pred, _, h, A_pred, h1 = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')  # 与概率p比？
        re_loss = F.mse_loss(x_bar, data)
        graph_loss = F.mse_loss(h, data)
        re_graphloss = F.binary_cross_entropy(A_pred.view(-1), adj.to_dense().view(-1))
        # , weight = weight_tensor

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss+0.01 * re_graphloss + 0.01 * graph_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


from warnings import simplefilter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='dblp')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    simplefilter(action='ignore', category=FutureWarning)

    args.pretrain_path = 'data/{}.pkl'.format(args.name)
    dataset = load_data(args.name)

    if args.name == 'usps':
        args.k=10
        args.n_clusters = 10
        args.n_input = 256

    if args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561

    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870

    if args.name == 'dblp':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 4
        args.n_input = 334

    if args.name == 'cite':
        args.lr = 5e-5
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703

    print(args)
    train_sdcn(dataset)
