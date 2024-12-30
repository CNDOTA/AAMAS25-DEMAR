import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EQMixer(nn.Module):
    def __init__(self, args):
        super(EQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1_list = nn.ModuleList([nn.Linear(self.state_dim, self.embed_dim * self.n_agents) for _ in range(self.args.mixer_N)])
            self.hyper_w_final_list = nn.ModuleList([nn.Linear(self.state_dim, self.embed_dim) for _ in range(self.args.mixer_N)])
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1_list = nn.ModuleList([nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                                               nn.ReLU(),
                                                               nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
                                                 for _ in range(self.args.mixer_N)])
            self.hyper_w_final_list = nn.ModuleList([nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                                                   nn.ReLU(),
                                                                   nn.Linear(hypernet_embed, self.embed_dim))
                                                     for _ in range(self.args.mixer_N)])
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1_list = nn.ModuleList([nn.Linear(self.state_dim, self.embed_dim) for _ in range(self.args.mixer_N)])

        # V(s) instead of a bias for the last layers
        self.V_list = nn.ModuleList([nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                                   nn.ReLU(),
                                                   nn.Linear(self.embed_dim, 1))
                                     for _ in range(self.args.mixer_N)])

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        el = agent_qs.size(1)

        if self.args.env == 'sc2':
            # FIXME 0721
            noise = th.rand(states.shape, device=self.args.device) * 0.02
            # noise = torch.normal(mean=0.0, std=0.02, size=inputs.shape, device=self.args.device)
            states += noise

        states = states.reshape(-1, self.state_dim)  # (bs * el, state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # (bs * el, 1, n_agents)
        q_tot_list = []
        hyper_l1_loss = 0
        v_list = []
        for i in range(self.args.mixer_N):
            # First layer
            w1 = th.abs(self.hyper_w_1_list[i](states))
            b1 = self.hyper_b_1_list[i](states)
            w1 = w1.view(-1, self.n_agents, self.embed_dim)
            b1 = b1.view(-1, 1, self.embed_dim)
            hidden = F.elu(th.bmm(agent_qs, w1) + b1)  # (bs*el, 1, emb)
            # Second layer
            w_final = th.abs(self.hyper_w_final_list[i](states))
            w_final = w_final.view(-1, self.embed_dim, 1)  # (bs, emb, 1)
            # State-dependent bias
            v = self.V_list[i](states).view(-1, 1, 1)  # (bs*el, 1, 1)
            # Compute final output
            y = th.bmm(hidden, w_final) + v  # (bs*el, 1, 1)
            # Reshape and return
            q_tot = y.view(bs, -1, 1)  # (bs, el, 1)
            q_tot_list.append(q_tot)
            # print((th.sum(w1) / (bs * el), th.sum(w_final) / (bs * el), th.sum(th.abs(b1) / (bs * el))))
            hyper_l1_loss += th.sum(w1) + th.sum(w_final) + th.sum(th.abs(b1)) + th.sum(th.abs(v))  # w1 >= 0, w2 >= 0
            v_list.append(v.view(bs, -1, 1))
        q_tot_list = th.cat(q_tot_list, dim=2)  # (bs, el, N)
        hyper_l1_loss = hyper_l1_loss / (self.args.mixer_N * bs * el)  # scaled
        # hyper_l1_loss = hyper_l1_loss / (self.args.mixer_N * bs * el * self.n_agents)  # scaled
        v_list = th.cat(v_list, dim=2)
        return q_tot_list, hyper_l1_loss, v_list

