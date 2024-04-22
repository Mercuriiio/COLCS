import torch
import torch.nn as nn
from random import sample
from torch.nn.modules.linear import Linear
import torchvision.models as models
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def full_block(in_features, out_features, p_drop=0.0):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        #nn.LayerNorm(out_features),
        nn.ReLU(),
        nn.Dropout(p=p_drop),
    )


class MLPEncoder(nn.Module):

    def __init__(self, num_genes=10000, num_hiddens=128, p_drop=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            full_block(num_genes, 1024, p_drop),
            full_block(1024, num_hiddens, p_drop),
            # add one block for features
        )

    def forward(self, x):

        x = self.encoder(x)

        return x


class Classification_model(nn.Module):
    def __init__(self, 
                 num_genes,
                 num_hiddens,
                 p_drop,
                 num_clusters):
        super().__init__()
        self.encoder = MLPEncoder(num_genes, num_hiddens, p_drop)
        self.classification = full_block(num_hiddens, num_clusters)

    def forward(self, x):
        feat = self.encoder(x)
        x = self.classification(feat)
        return feat, x


class MoCo(nn.Module):
    def __init__(self, base_encoder, num_genes=10000,  dim=16, r=512, m=0.999, T=0.2, n=6):
        super(MoCo, self).__init__()

        self.r = r
        self.m = m
        self.T = T

        self.encoder_q = base_encoder(num_genes=num_genes, num_hiddens=dim)
        self.encoder_k = base_encoder(num_genes=num_genes, num_hiddens=dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.cluster_layer = Parameter(torch.Tensor(n, dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.alpha = 1.0

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.r % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.r

        self.queue_ptr[0] = ptr


    def forward(self, im_q, im_k=None, is_eval=False, cluster_result=None, index=None):
        
        if is_eval:
            k = self.encoder_k(im_q)  
            k = nn.functional.normalize(k, dim=1)            
            return k
        
        # compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()

            k = self.encoder_k(im_k)
            # print(k.shape)
            k = nn.functional.normalize(k, dim=1)

        # compute query features
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        # cluster
        pq = 1.0 / (1.0 + torch.sum(
            torch.pow(q.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        pq = pq.pow((self.alpha + 1.0) / 2.0)
        pq = (pq.t() / torch.sum(pq, 1)).t()

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # 矩阵乘法
        
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long)  #.cuda()

        self._dequeue_and_enqueue(k)

        return logits, labels, q, pq
