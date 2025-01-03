import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from run import run

import argparse
import time, datetime
import tempfile
import random

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    random.seed(config['seed'])
    config['env_args']['seed'] = config["seed"]

    if config['env'] == 'mpe_env':
        assert config['scenario_name'] in ['simple_tag', 'simple_world', 'simple_adversary', 'simple_crypto']

        config['target_update_interval'] = 800

        # if config['scenario_name'] in ['simple_tag', 'simple_world']:
        #     config['res_lambda'] = 0.05
        # elif config['scenario_name'] in ['simple_adversary']:
        #     config['res_lambda'] = 0.5
        # elif config['scenario_name'] in ['simple_crypto']:
        #     config['res_lambda'] = 0.01
    else:
        # assert config['env_args']['map_name'] in ['2s3z', '3s5z', '2c_vs_64zg', 'MMM2']

        config['res_beta'] = 5.0
        # if config['env_args']['map_name'] in ['2s3z', '3s5z', '2c_vs_64zg']:
        #     config['res_lambda'] = 0.05
        # elif config['env_args']['map_name'] in ['MMM2']:
        #     config['res_lambda'] = 0.01

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    if config_dict['env'] == 'mpe_env':
        file_obs_path = os.path.join(results_path, config_dict['name'], "sacred")
    elif config_dict['env'] == 'sc2':
        file_obs_path = os.path.join(results_path, config_dict['name'], "sacred")
    else:
        raise NotImplementedError('Unknown Env.')
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

