# CDNMF

import os
import pickle
import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.device = config['device']
        self.net_shape = config['net_shape']
        self.att_shape = config['att_shape']
        self.net_input_dim = config['net_input_dim']
        self.att_input_dim = config['att_input_dim']
        self.is_init = config['is_init']
        self.pretrain_params_path = config['pretrain_params_path']
        self.tau = config['tau']
        self.conc = config['conc']
        self.negc = config['negc']
        self.rec = config['rec']
        self.r = config['r']
        self.model_path = config['model_path']

        self.fc1 = torch.nn.Linear(self.net_shape[-1], self.net_shape[1])
        self.fc2 = torch.nn.Linear(self.net_shape[1], self.net_shape[0])

        self.fc3 = torch.nn.Linear(self.att_shape[-1], self.net_shape[1])
        self.fc4 = torch.nn.Linear(self.net_shape[1], self.net_shape[0])

        self.U = torch.nn.ParameterDict({})
        self.V = torch.nn.ParameterDict({})

        if os.path.isfile(self.pretrain_params_path):
            with open(self.pretrain_params_path, 'rb') as handle:
                self.U_init, self.V_init = pickle.load(handle)

        if self.is_init:
            module = 'net'
            for i in range(len(self.net_shape)):
                name = module + str(i)
                self.U[name] = torch.nn.Parameter(torch.tensor(self.U_init[name], dtype=torch.float32))
            self.V[name] = torch.nn.Parameter(torch.tensor(self.V_init[name], dtype=torch.float32))

            module = 'att'
            for i in range(len(self.att_shape)):
                name = module + str(i)
                self.U[name] = torch.nn.Parameter(torch.tensor(self.U_init[name], dtype=torch.float32))
            self.V[name] = torch.nn.Parameter(torch.tensor(self.V_init[name], dtype=torch.float32))
        else:
            module = 'net'
            for i in range(len(self.net_shape)):
                name = module + str(i)
                self.U[name] = torch.nn.Parameter(torch.rand_like(torch.tensor(self.U_init[name], dtype=torch.float32)))
            self.V[name] = torch.nn.Parameter(torch.rand_like(torch.tensor(self.V_init[name], dtype=torch.float32)))

            module = 'att'
            for i in range(len(self.att_shape)):
                name = module + str(i)
                self.U[name] = torch.nn.Parameter(torch.rand_like(torch.tensor(self.U_init[name]), dtype=torch.float32))
            self.V[name] = torch.nn.Parameter(torch.rand_like(torch.tensor(self.V_init[name], dtype=torch.float32)))

    def projection1(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z.t()))
        return self.fc2(z)

    def projection2(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc3(z.t()))
        return self.fc4(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, chunk_size: int = 4096):
        # Same as the paper loss, but never materializes an N x N similarity matrix.
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        n = z1.size(0)
        tau = self.tau

        self.index_net = torch.argmax(self.V1, dim=0).long()
        between_diag = torch.exp((z1 * z2).sum(dim=1) / tau)

        refl_sum = z1.new_zeros(n)
        refl_pos_sum = z1.new_zeros(n)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            refl = torch.exp(torch.mm(z1, z1[start:end].t()) / tau)
            refl_sum = refl_sum + refl.sum(dim=1)
            same = self.index_net.unsqueeze(1).eq(self.index_net[start:end].unsqueeze(0))
            refl_pos_sum = refl_pos_sum + (refl * same).sum(dim=1)
        return -torch.log(between_diag / (refl_sum - refl_pos_sum + between_diag))

    def contra_loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True):
        h1 = self.projection1(z1)
        h2 = self.projection2(z2)

        ret = self.semi_loss(h1, h2)
        ret = ret.mean() if mean else ret.sum()

        return ret

    def forward(self):
        self.V1 = self.V['net' + str(len(self.net_shape) - 1)]
        self.V2 = self.V['att' + str(len(self.att_shape) - 1)]
        return self.V1

    def _factor_pair(self, prefix, n_layers):
        C = self.U[prefix + '0']
        for i in range(1, n_layers):
            C = torch.mm(C, self.U[prefix + str(i)])
        V = self.V[prefix + str(n_layers - 1)]
        return C, V

    def _recon_fro(self, target, C, V):
        # ||target - C@V||_F^2 without forming C@V when that would be N x N
        inner = torch.trace(torch.mm(C.t(), torch.mm(target, V.t())))
        pnorm = torch.trace(torch.mm(torch.mm(C.t(), C), torch.mm(V, V.t())))
        tnorm = torch.square(torch.norm(target))
        return tnorm + pnorm - 2 * inner

    def _trace_lap(self, Z, A):
        # trace(Z (D-A) Z.T) without forming the Laplacian
        deg = torch.sum(A, dim=1)
        return (Z.pow(2).sum(0) * deg).sum() - torch.sum(torch.mm(Z, A) * Z)

    def loss(self, graph):

        A = graph.A
        X = graph.X.T

        C1, V1 = self._factor_pair('net', len(self.net_shape))
        loss1 = self._recon_fro(A, C1, V1)

        C2, V2 = self._factor_pair('att', len(self.att_shape))
        loss2 = self._recon_fro(X, C2, V2)

        loss3 = self.contra_loss(self.V1, self.V2)

        i = len(self.net_shape) - 1
        loss4 = self._trace_lap(self.V['net' + str(i)], A)
        i = len(self.att_shape) - 1
        loss4 = loss4 + self._trace_lap(self.V['att' + str(i)], A)

        loss5 = 0
        for i in range(len(self.net_shape)):
            zero1 = torch.zeros_like(self.U['net' + str(i)])
            X1 = torch.where(self.U['net' + str(i)] > 0, zero1, self.U['net' + str(i)])
            loss5 = loss5 + torch.square(torch.norm(X1))
        zero1 = torch.zeros_like(self.V['net' + str(i)])
        X1 = torch.where(self.V['net' + str(i)] > 0, zero1, self.V['net' + str(i)])
        loss5 = loss5 + torch.square(torch.norm(X1))

        for i in range(len(self.att_shape)):
            zero2 = torch.zeros_like(self.U['att' + str(i)])
            X2 = torch.where(self.U['att' + str(i)] > 0, zero2, self.U['att' + str(i)])
            loss5 = loss5 + torch.square(torch.norm(X2))
        i = len(self.att_shape) - 1
        zero2 = torch.zeros_like(self.V['att' + str(i)])
        X2 = torch.where(self.V['att' + str(i)] > 0, zero2, self.V['att' + str(i)])
        loss5 = loss5 + torch.square(torch.norm(X2))

        loss = self.rec*(loss1 + loss2) + self.conc*loss3 + self.r*loss4 + self.negc*loss5

        return loss, loss1, loss2, loss3, loss4, loss5
