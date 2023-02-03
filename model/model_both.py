import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from model.model_utils import *
from model.model_adp import STAttention_Adp
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Model_Both(nn.Module):
    def __init__(self, SE, args, window_size = 3, T = 12, N=None, g=None):
        super(Model_Both, self).__init__()
        L = args.L
        K = args.K
        d = args.d
        D = K * d

        self.num_his = args.num_his
        self.SE = SE.to(device)
        emb_dim = SE.shape[1]
        self.STEmbedding = STEmbedding(D, emb_dim=emb_dim).to(device)

        self.STAttBlock_1 = nn.ModuleList([ST_Layer(K, d, T=T, window_size = window_size,N=N,g=g) for _ in range(L)])
        self.STAttBlock_2 = nn.ModuleList([ST_Layer(K, d, T=T, window_size = window_size,N=N,g=g) for _ in range(L)])
        self.transformAttention = TransformAttention(K, d)

        self.mlp_1 = CONVs(input_dims=[1, D], units=[D, D], activations=[F.relu, None])
        self.mlp_2 = CONVs(input_dims=[D, D], units=[D, 1], activations=[F.relu, None])

    def forward(self, X, TE):
        # input
        X = torch.unsqueeze(X, -1)
        X = self.mlp_1(X)
        # STE
        STE = self.STEmbedding(self.SE, TE)
        STE_his = STE[:, :self.num_his]
        STE_pred = STE[:, self.num_his:]
        # encoder
        for net in self.STAttBlock_1:
            X = net(X, STE_his)
        X = self.transformAttention(X, STE_his, STE_pred)
        # decoder
        for net in self.STAttBlock_2:
            X = net(X, STE_pred)
        X = self.mlp_2(X)
        del STE, STE_his, STE_pred

        return torch.squeeze(X, 3)

class ST_Layer(nn.Module):
    def __init__(self, K, d, T=12, window_size =5, N=None,g=None ):
        super(ST_Layer, self).__init__()
        self.stAtt_adp = STAttention_Adp(K, d, T=T, window_size=window_size, N=N)
        self.stAtt_loc = STAttention(K, d, T=T, window_size=window_size, N=N,g=g)
        # self.tAtt = TAttention(K, d, bn_decay, mask=mask)
        self.mlp = PostProcess(K * d)
        self.fusion = GatedFusion(K * d)
    
    def forward(self, X, STE):
        HS_adp = self.stAtt_adp(X, STE)
        HS_loc = self.stAtt_loc(X, STE)
        H = self.fusion(HS_adp, HS_loc) 
        return H

class STAttention_Adp(nn.Module):
    def __init__(self, K, d, T=12, window_size =5, N=None):
        super(STAttention_Adp, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.window = window_size
        self.T = T
        self.N = N
 
        self.dropout = 0.1
        self.FC_q  = nn.Linear(2*D, D)
        self.FC_k  = nn.Linear(2*D, D)
        self.FC_v  = nn.Linear(2*D, D)

        self.nodevec1 =nn.Parameter(torch.randn(N, 20).cuda(), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(20, N).cuda(), requires_grad=True)

        self.attn_output =  TransformerSelfOutput(D,D)
        self.shift_list = self.get_shift_list()

    def get_shift_list(self):
        idxs = np.arange(self.T)
        window_size = self.window
        window_list = np.arange(-(window_size-1)//2,(window_size-1)//2+1,1)
        shift_list = []
        for i in window_list:
            tmp = idxs+i
            tmp[tmp<0] = tmp[tmp<0] + window_size
            tmp[tmp>(self.T-1)] = tmp[tmp>(self.T-1)] - window_size
            shift_list.append(tmp)
        shift_list = np.array(shift_list)
        return shift_list

    def get_adp_graph(self, max_num_neigh = 40):
        adp_A = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        threshold = 1/self.N
        tmp,_ = torch.kthvalue(-1*adp_A, max_num_neigh + 1,dim=1,keepdim=True)
        bin_mask = (torch.logical_and((adp_A > threshold), (adp_A > -tmp)).type_as(adp_A) - adp_A).detach() + adp_A
        adp_A = adp_A*bin_mask
        idxs= torch.nonzero(adp_A)
        src,dst = idxs[:,0],idxs[:,1]
        adp_g = dgl.graph((src, dst)).to("cuda")
        # adp_g.edata['weight'] = adp_A[src,dst].cuda()
        return adp_g

    def forward(self, X, STE):
        X_STE = torch.cat((X, STE), dim=-1)
        query = self.FC_q(X_STE)
        key = self.FC_k(X_STE)    
        value = self.FC_v(X_STE)
        B = query.shape[0]
        T = query.shape[1]
        N = query.shape[2]
        hdim = query.shape[3]//self.K
        query = query.view(B, T, N, self.K, hdim)
        key   =  key.view (B, T, N, self.K, hdim)
        value = value.view(B, T, N, self.K, hdim)
        query = query.permute(2,0,1,3,4) 
        value = value.permute(2,0,1,3,4)
        key   = key.permute(2,0,1,3,4)
        
        g = self.get_adp_graph()
        g = g.local_var()
        res = 0
        for ti in range(len(self.shift_list)):
            g.ndata['q'] =  query/(hdim**0.5)
            g.ndata['k'] =  key[:,:,self.shift_list[ti],:,:]
            g.ndata['v'] =  value[:,:,self.shift_list[ti],:,:]
            g.apply_edges(fn.u_dot_v('k', 'q', 'score')) 
            # g.apply_edges(mask_attention_score)
            e = g.edata.pop('score') 
            g.edata['score'] = edge_softmax(g, e)
            g.edata['score']= nn.functional.dropout(g.edata['score'], p=self.dropout, training=self.training)
            g.update_all(fn.u_mul_e('v', 'score', 'm'), fn.sum('m', 'h'))
            attn_output = g.ndata['h'] 
            attn_output = attn_output.permute(1,2,0,3,4)
            attn_output = attn_output.reshape(B,T,N,self.K * hdim)
            res += attn_output
        res /= len(self.shift_list)
        attn_output = self.attn_output(res, X)
        return attn_output


class STAttention(nn.Module):
    def __init__(self, K, d, T=12, window_size =5, N=None, g=None):
        super(STAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.window = window_size
        self.T = T
        self.N = N
        self.g = g.to(device)

        self.dropout = 0.1
        self.FC_q  = nn.Linear(2*D, D)
        self.FC_k  = nn.Linear(2*D, D)
        self.FC_v  = nn.Linear(2*D, D)
 
        self.nodevec1 =nn.Parameter(torch.randn(N, 20).cuda(), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(20, N).cuda(), requires_grad=True)

        self.attn_output =  TransformerSelfOutput(D,D)
        self.shift_list = self.get_shift_list()

    def get_shift_list(self):
        idxs = np.arange(self.T)
        window_size = self.window
        window_list = np.arange(-(window_size-1)//2,(window_size-1)//2+1,1)
        shift_list = []
        for i in window_list:
            tmp = idxs+i
            tmp[tmp<0] = tmp[tmp<0] + window_size
            tmp[tmp>(self.T-1)] = tmp[tmp>(self.T-1)] - window_size
            shift_list.append(tmp)
        shift_list = np.array(shift_list)
        return shift_list


    def forward(self, X, STE):
        X_STE = torch.cat((X, STE), dim=-1)
        query = self.FC_q(X_STE) 
        key = self.FC_k(X_STE)    
        value = self.FC_v(X_STE)
        B = query.shape[0]
        T = query.shape[1]
        N = query.shape[2]
        hdim = query.shape[3]//self.K
        query = query.view(B, T, N, self.K, hdim)
        key   =  key.view (B, T, N, self.K, hdim)
        value = value.view(B, T, N, self.K, hdim)
        query = query.permute(2,0,1,3,4)
        value = value.permute(2,0,1,3,4)
        key   = key.permute(2,0,1,3,4)
        
        g = self.g.local_var()
        res = 0
        for ti in range(len(self.shift_list)):
            g.ndata['q'] =  query/(hdim**0.5)
            g.ndata['k'] =  key[:,:,self.shift_list[ti],:,:]
            g.ndata['v'] =  value[:,:,self.shift_list[ti],:,:] 
            g.apply_edges(fn.u_dot_v('k', 'q', 'score'))
            # g.apply_edges(mask_attention_score)
            e = g.edata.pop('score') 
            g.edata['score'] = edge_softmax(g, e)
            g.edata['score']= nn.functional.dropout(g.edata['score'], p=self.dropout, training=self.training)
            g.update_all(fn.u_mul_e('v', 'score', 'm'), fn.sum('m', 'h'))
            attn_output = g.ndata['h']
            attn_output = attn_output.permute(1,2,0,3,4)
            attn_output = attn_output.reshape(B,T,N,self.K * hdim)
            res += attn_output
        res /= len(self.shift_list)
        attn_output = self.attn_output(res, X)
        return attn_output