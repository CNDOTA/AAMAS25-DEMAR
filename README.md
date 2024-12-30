# Dual Ensembled Multiagent Q-Learning with Hypernet Regularizer

This repository is the implementation of [Dual Ensembled Multiagent Q-Learning with Hypernet Regularizer](https://openreview.net/forum?id=6bAkCauS3N) in AAMAS 2025. This codebase is based on the open-source [RES](https://github.com/ling-pan/RES), [PyMARL](https://github.com/oxwhirl/pymarl) framework and [maddpg-pytorch](https://github.com/shariqiqbal2810/maddpg-pytorch), and please refer to that repo for more documentation.

## Citing
If you used this code in your research or found it helpful, please consider citing our paper:
Bibtex:
```
@inproceedings{
yang2024dual,
title={Dual Ensembled Multiagent Q-Learning with Hypernet Regularizer},
author={Yaodong Yang, Guangyong Chen, Hongyao Tang, Furui Liu, Danruo Deng, and Pheng-Ann Heng},
booktitle={The 24th International Conference on Autonomous Agents and Multi-Agent Systems},
year={2024},
url={https://openreview.net/forum?id=6bAkCauS3N}
}
```

## Requirements
- PyMARL: Please check the [PyMARL](https://github.com/oxwhirl/pymarl) repo for more details about the environment.
- Multi-agent Particle Environments: in envs/multiagent-particle-envs and install it by `pip install -e .`
- SMAC: Please check the [SMAC](https://github.com/oxwhirl/smac) repo for more details about the environment. Note that for all SMAC experiments we used the latest version SC2.4.10. The results reported in the SMAC paper (https://arxiv.org/abs/1902.04043) use SC2.4.6.2.69232. Performance is not always comparable across versions.

## Usage
Please follow the instructions below to replicate the results in the paper. Hyperparameters can be configured in config files.

### Multi-Agent Particle Environments
Please use the corresponding hyperparameters provided in our paper to run each task.
Please specify hyperparameters in the "/config/alg/demix.yaml".
For example, if you want to run demar on simple_tag, please use the following command with the corresponding "demix.yaml"
where hyperparameters are configured based on the below table (Appendix E of our paper).
```
python src/main.py --config=demix --env-config=mpe_env with scenario_name=simple_tag seed=1 > out_demar_pp_1.log 2>&1 &
```
| DEMAR            | simple tag | simple world | simple adversary |
|------------------|------------|--------------|------------------|
| $H$              | 3          | 10           | 10               |
| $N_{\mathbb{H}}$ | 3          | 6            | 4                |
| $K$              | 1          | 1            | 10               |
| $N_{\mathbb{K}}$ | 1          | 1            | 4                |
| $\alpha_{reg}$   | 0.002      | 0.02         | 0.05             |

### StarCraft Multi-Agent Challenge
Please use the corresponding hyperparameters provided in our paper to run each task.
Please specify hyperparameters in the "/config/alg/demix.yaml".
For example, if you want to run demar on 5m_vs_6m, please use the following command with the corresponding "demix.yaml"
where hyperparameters are configured based on the below table (Appendix E of our paper).
```
python src/main.py --config=demix --env-config=sc2 with env_args.map_name=5m_vs_6m env_args.seed=1 > out_demar_5m_vs_6m_1.log 2>&1 &
```
| DEMAR            | 5m\_vs\_6m | 2s3z  | 3s5z  | 10m\_vs\_11m |
|------------------|------------|-------|-------|--------------|
| $H$              | 3          | 3     | 10    | 4            |
| $N_{\mathbb{H}}$ | 2          | 2     | 9     | 3            |
| $K$              | 1          | 1     | 1     | 1            |
| $N_{\mathbb{K}}$ | 1          | 1     | 1     | 1            |
| $\alpha_{reg}$   | 0.002      | 0.002 | 0.001 | 0.01         |
