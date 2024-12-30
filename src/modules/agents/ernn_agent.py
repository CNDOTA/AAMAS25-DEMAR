import torch
import torch.nn as nn
import torch.nn.functional as F


class ERNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ERNNAgent, self).__init__()
        self.args = args

        self.fc1_list = nn.ModuleList([nn.Linear(input_shape, args.rnn_hidden_dim) for _ in range(self.args.agent_N)])
        self.rnn_list = nn.ModuleList([nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim) for _ in range(self.args.agent_N)])
        self.fc2_list = nn.ModuleList([nn.Linear(args.rnn_hidden_dim, args.n_actions) for _ in range(self.args.agent_N)])

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1_list[0].weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state_list):

        if self.args.env == 'sc2':
            # FIXME 0721: add noise into inputs
            noise = torch.rand(inputs.shape, device=self.args.device) * 0.02
            # noise = torch.normal(mean=0.0, std=0.02, size=inputs.shape, device=self.args.device)
            inputs += noise

        q_list, h_list = [], []
        for i in range(self.args.agent_N):
            x = F.relu(self.fc1_list[i](inputs))
            h_in = hidden_state_list[i].reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn_list[i](x, h_in)
            q = self.fc2_list[i](h)  # (bs, n_agent, n_actions)
            q_list.append(q)
            h_list.append(h)
        q_list = torch.stack(q_list, dim=-1)  # (bs, n_agent, n_actions, N)
        return q_list, h_list
