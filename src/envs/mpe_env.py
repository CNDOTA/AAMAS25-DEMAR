from envs.multiagentenv import MultiAgentEnv

import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
try:
    # from multiagent.multi_discrete import MultiDiscrete
    from gym.spaces import MultiDiscrete
except:
    print ('multiagent is not installed')
    
def int2onehot(ids, onehot_dim):
    import torch
    onehot_tensors = torch.zeros(ids.shape[0], onehot_dim)
    onehot_tensors.scatter_(dim=1, index=ids.cpu().view(-1, 1), value=1.)
    return onehot_tensors

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MPEEnv(MultiAgentEnv):
    metadata = {'render.modes' : ['human', 'rgb_array']}

    def __init__(
        self, world, 
        reset_callback=None, reward_callback=None, observation_callback=None, info_callback=None, done_callback=None, post_step_callback=None, shared_viewer=True, 
        discrete_action=True,
        episode_limit=None,
        **kwargs
    ):
        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.post_step_callback = post_step_callback
        # environment parameters
        self.discrete_action_space = discrete_action
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = False
        self.time = 0

        assert self.world.scenario_name in ['simple_tag', 'simple_world', 'simple_adversary', 'simple_crypto']
        if self.world.scenario_name in ['simple_tag', 'simple_world']:
            self.num_cooperating_agents, self.num_opponent_agents = self.world.num_adversaries, self.world.num_good_agents
        elif self.world.scenario_name in ['simple_adversary', 'simple_crypto']:
            self.num_cooperating_agents, self.num_opponent_agents = self.world.num_good_agents, self.world.num_adversaries
        self.n_agents = self.num_cooperating_agents
        self.num_landmarks = self.world.num_landmarks

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent_idx, agent in enumerate(self.agents):
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,))
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            c_action_space = spaces.Discrete(world.dim_c)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    # from multiagent.multi_discrete import MultiDiscrete
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,)))
            agent.action.c = np.zeros(self.world.dim_c)

        # TODO: only supports same size of the action space currently
        for agent_idx in range(1, len(self.action_space)):
            assert self.action_space[agent_idx].n == self.action_space[0].n

        if self.world.scenario_name in ['simple_tag', 'simple_world']:
            pivot_agent_idx = 0
        elif self.world.scenario_name in ['simple_adversary', 'simple_crypto']:
            pivot_agent_idx = -1
        self.obs_size = self.observation_space[pivot_agent_idx].shape[0]
        self.n_actions = self.action_space[pivot_agent_idx].n

        self.state_size = self.get_state().shape[0]

        self.episode_limit = episode_limit

        self.steps = 0
        self.truncate_episodes = True

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    ##################### getter functions #####################
    def get_total_actions(self):
        return self.n_actions

    def get_obs_size(self):
        return self.obs_size

    def get_state_size(self):
        return self.state_size

    def get_env_info(self):
        info = MultiAgentEnv.get_env_info(self)
        return info

    def get_state(self):
        return np.concatenate(self.get_obs())

    def get_obs(self):
        """ Returns all agent observations in a list """
        obs_n = []
        for agent_idx, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
        return obs_n

    def get_avail_actions(self):
        avail_actions = [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]
        return avail_actions
    
    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space

    def get_agent_num_infos(self):
        return self.n, self.num_cooperating_agents, self.num_opponent_agents
    ##################### getter functions #####################

    def _seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def step(self, action_n):
        if self.discrete_action_space:
            action_n = int2onehot(action_n, self.n_actions)

        obs_n = []
        reward_n = []
        done_n = []
        # info_n = {'n': []}
        info_n = {}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n
        
        self.steps += 1
        if self.steps >= self.episode_limit:
            info_n['episode_limit'] = self.truncate_episodes
            done_n = [True for _ in range(len(self.agents))]
        else:
            info_n['episode_limit'] = False

        if self.post_step_callback is not None:
            self.post_step_callback(self.world)
        
        return reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()

        self.steps = 0

        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def close(self):
        import pygame
        pygame.quit()
        print("Closing Multi-Agent Navigation")

    # render environment
    def render(self, mode='human', close=True):
        close = False
        if close:
            # close any existic renderers
            for i,viewer in enumerate(self.viewers):
                if viewer is not None:
                    viewer.close()
                self.viewers[i] = None
            return []

        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            self.comm_geoms = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                entity_comm_geoms = []
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                    if not entity.silent:
                        dim_c = self.world.dim_c
                        # make circles to represent communication
                        for ci in range(dim_c):
                            comm = rendering.make_circle(entity.size / dim_c)
                            comm.set_color(1, 1, 1)
                            comm.add_attr(xform)
                            offset = rendering.Transform()
                            comm_size = (entity.size / dim_c)
                            offset.set_translation(ci * comm_size * 2 -
                                                   entity.size + comm_size, 0)
                            comm.add_attr(offset)
                            entity_comm_geoms.append(comm)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
                self.comm_geoms.append(entity_comm_geoms)
            for wall in self.world.walls:
                corners = ((wall.axis_pos - 0.5 * wall.width, wall.endpoints[0]),
                           (wall.axis_pos - 0.5 * wall.width, wall.endpoints[1]),
                           (wall.axis_pos + 0.5 * wall.width, wall.endpoints[1]),
                           (wall.axis_pos + 0.5 * wall.width, wall.endpoints[0]))
                if wall.orient == 'H':
                    corners = tuple(c[::-1] for c in corners)
                geom = rendering.make_polygon(corners)
                if wall.hard:
                    geom.set_color(*wall.color)
                else:
                    geom.set_color(*wall.color, alpha=0.5)
                self.render_geoms.append(geom)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
                for entity_comm_geoms in self.comm_geoms:
                    for geom in entity_comm_geoms:
                        viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                if 'agent' in entity.name:
                    self.render_geoms[e].set_color(*entity.color, alpha=0.5)
                    if not entity.silent:
                        for ci in range(self.world.dim_c):
                            color = 1 - entity.state.c[ci]
                            self.comm_geoms[e][ci].set_color(color, color, color)
                else:
                    self.render_geoms[e].set_color(*entity.color)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx