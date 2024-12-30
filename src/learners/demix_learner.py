import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.eqmix import EQMixer
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop
import numpy as np


class DEMLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == 'eqmix':
                self.mixer = EQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))

            mixer_param_num = sum(param.numel() for param in self.mixer.parameters())
            print('{} mixer parameter number is {}'.format(self.args.mixer, mixer_param_num))

        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time, (bs, el, n_agent, n_action, agent_N)

        mac_out = mac_out.mean(dim=-1, keepdim=False)  # (bs, el, n_agent, n_action)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        agent_utility = chosen_action_qvals.detach()
        agent_mask = mask.expand_as(agent_utility).detach()

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time, (bs, el, n_agent, n_action, agent_N)

        agent_sample_idxs = np.random.choice(self.args.agent_N, self.args.agent_M, replace=False)
        target_mac_out = target_mac_out[:, :, :, :, agent_sample_idxs].min(dim=-1, keepdim=False)[0]

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]  # (bs, ts, n_agents, action)
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # mix, (bs, el, N)
        # chosen_action_qvals, hyper_l1_loss, vs = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        # 20230826: record the gradients
        qtot_vals, hyper_l1_loss, vs = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        qtot_qi_gradients = th.autograd.grad(qtot_vals.sum(), chosen_action_qvals, retain_graph=True)[0]
        chosen_action_qvals = qtot_vals

        target_max_qvals, _, target_vs = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        # sample M from N Q_target
        sample_idxs = np.random.choice(self.args.mixer_N, self.args.mixer_M, replace=False)
        # r + gamma * min_{i in M} Q_target_next
        # rewards: (bs, el, 1); target_max_qvals: (bs, el, N); terminated: (bs, el, 1)
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals[:, :, sample_idxs].min(dim=2, keepdim=True)[0]
        # Td-error, chosen_action_qvals: (bs, el, N), targets: (bs, el, 1)
        td_error = chosen_action_qvals - targets.detach()  # (bs, el, N)
        # mask: (bs, el, 1)
        mask = mask.expand_as(td_error)  # (bs, el, N)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask  # (bs, el, N)
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum() + self.args.hyper_alpha * hyper_l1_loss  # + 0.01 * (mixer_l1_loss / self.args.mixer_N)

        if self.args.v_reg:
            future_episode_return = batch["future_discounted_return"][:, :-1]
            # chosen_action_qvals: (bs, el, N); future_episode_return: (bs, el, 1)
            q_return_diff = chosen_action_qvals - future_episode_return.detach()
            v_l2 = ((q_return_diff * mask) ** 2).sum() / mask.sum()  # mask: (bs, el, N)
            loss += self.args.red_lambda * v_l2

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("hyper_loss", hyper_l1_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            agent_mask_elems = agent_mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("agent_utility", ((agent_utility * agent_mask).sum().item() / (agent_mask_elems * self.args.n_agents)), t_env)
            self.logger.log_stat("qtot_qi_gradient", ((qtot_qi_gradients * agent_mask).sum().item() / (agent_mask_elems * self.args.n_agents)), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("v_taken_mean", (vs * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_v_mean", (target_vs * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
