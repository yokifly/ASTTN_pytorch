import torch
import torch.nn as nn
import torch.nn.functional as F
import math
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size=[1, 1], stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)
    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)

class CONVs(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay=0.1, use_bias=True):
        super(CONVs, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dims, output_dims, activation=F.relu):
        super(MLP, self).__init__()
        self.activation = activation
        self.mlp = nn.Linear(input_dims, output_dims)
        self.layer_norm = nn.LayerNorm(output_dims)
        torch.nn.init.xavier_uniform_(self.mlp.weight)
        torch.nn.init.zeros_(self.mlp.bias)

    def forward(self, x):
        x = self.mlp(x)
        x = self.layer_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x

class MLPs(nn.Module):
    def __init__(self, input_dims, units, activations):
        super(MLPs, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.mlps = nn.ModuleList([MLP(
            input_dims=input_dim, output_dims=num_unit, activation=activation
            ) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for mlp in self.mlps:
            x = mlp(x)
        return x


class STEmbedding(nn.Module):
    def __init__(self, D, emb_dim):
        super(STEmbedding, self).__init__()
        self.mlp_se = CONVs(
            input_dims=[emb_dim, D], units=[D, D], activations=[F.relu, None])

        self.mlp_te = CONVs(
            input_dims=[295, D], units=[D, D], activations=[F.relu, None])  # input_dims = time step per day + days per week=288+7=295

    def forward(self, SE, TE, T=288):
        # spatial embedding
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.mlp_se(SE)
        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1).to(device)
        TE = TE.unsqueeze(dim=2)
        TE = self.mlp_te(TE)
        del dayofweek, timeofday
        return SE + TE

class GatedFusion(nn.Module):
    def __init__(self, D):
        super(GatedFusion, self).__init__()
        self.mlp_xs = CONVs(input_dims=D, units=D, activations=None)
        self.mlp_xt = CONVs(input_dims=D, units=D, activations=None)
        self.mlp_h = CONVs(input_dims=[D, D], units=[D, D], activations=[F.relu, None])

    def forward(self, HS, HT):
        XS = self.mlp_xs(HS)
        XT = self.mlp_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.mlp_h(H)
        del XS, XT, z
        return H

class TransformerSelfOutput(nn.Module):
    def __init__(self, insize, outsize,dropout = 0.1):
        super().__init__()
        self.dense = nn.Linear(insize, outsize)
        self.LayerNorm = nn.LayerNorm(outsize)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class TransformerIntermediate(nn.Module):
    def __init__(self, hidden_size,intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class PostProcess(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.intermediate = TransformerIntermediate(D,D)
        self.layer_output = TransformerSelfOutput(D,D)

    def forward(self, HS):
        H_int = self.intermediate(HS)
        out  = self.layer_output(H_int,HS)
        return out

class TransformAttention(nn.Module):
    def __init__(self, K, d):
        super(TransformAttention, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.mlp_q = MLPs(input_dims=D, units=D, activations=None)
        self.mlp_k = MLPs(input_dims=D, units=D, activations=None)
        self.mlp_v = MLPs(input_dims=D, units=D, activations=None)
        self.out = MLPs(input_dims=D, units=D, activations=F.relu)

    def forward(self, X, STE_his, STE_pred):
        batch_size = X.shape[0]
        query = self.mlp_q(STE_pred)
        key = self.mlp_k(STE_his)
        value = self.mlp_v(X)
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        out = torch.matmul(attention, value)
        out = out.permute(0, 2, 1, 3)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.out(out)
        return out