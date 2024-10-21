# Robust Multi-Agent Reinforcement Learning by Mutual Information Regularization

This repository contains the code for the MIR3 defense method as well as other baseline methods, such as MADDPG, M3DDPG, ROM-Q, ERNIE and ROMAX. It also includes implementations of different Multi-Agent Reinforcement Learning (MARL) environments on which we evaluate our results, such as SMAC and Multi-agent rendezvous.

## Supported Algorithms

* MARL training algorithms:
  
  - MADDPG
* MARL defense algorithms:
  * M3DDPG

  * ROM-Q

  * ERNIE

  * ROMAX
  
  * MIR3
  
* MARL attack algorithms:
  * Adversarial Policy


## Supported Environments

* SMAC
* Multi-agent rendezvous

## How to run the code

### Choose the Training Algorithm and Environment

The code uses different parameters for different algorithms and environments. The default parameters are located in `./configs`. Algorithm parameters are stored in YAML files named `{algorithm}_{env}.yaml` within the `./configs/alg` directory:

* `{algorithm}`:
  * MADDPG: maddpg
  * M3DDPG: m3ddpg
  * ROM-Q: romq
  * ERNIE: ernie
  * ROMAX: romax
  * MIR3: mir3
  * Adversarial policy: maddpg_traitor_mi
* `{env}`:
  * SMAC: smac
  * Multi-agent rendezvous: robot

For instance, parameters for MADDPG on SMAC environments are found in `./configs/alg/maddpg_smac.yaml`.

Environment parameters are saved in YAML files named `{env}.yaml` within the `./configs/env` directory:

* `{env}`:
  * SMAC for training: sc2
  * SMAC for attack: sc2_traitor
  * Multi-agent rendezvous: robot

For instance, parameters for SMAC environments during the attack phase are in `./configs/env/sc2_traitor.yaml`.

### Train the Agents and Save the models 

To train the agents, use the following command for example:

```bash
python -u main.py --alg mir3_smac --env sc2 --token train --map 4m_vs_3m --param_club 0.1 --seed 0
```

* `--alg`: the specified config file of the algorithm, which means that we use the default parameters of MIR3 in `./configs/algs/{alg}.yaml`. 
* `--env`: the specified config file of the environment, which means that we use the default parameters of sc2 for training in `./configs/envs/{env}.yaml`. 
* `--map`: the specified map name through the parameter `--map`. If the string specified in `--map` is none, the map name in the specified config file of the environment is used instead. 
* `--param_club`: the specified hyperparameter in penalizing mutual information. The default parameter of `--param_club` is saved in  `./configs/algs/mir3_smac.yaml`. 
*  `--token`:  the specified experiment name. The default parameter of `--token` is saved in  `./configs/default.yaml`.
* `--seed`:  the specified seed. The default parameter of `--seed` is saved in  `./configs/default.yaml`.

The models and the training datas are saved in the directories like:

```bash
models: ./results/{env}/{map}/none/{learner}/{token}/{seed}/models/{timestep}/
datas: ./results/{env}/{map}/none/{learner}/{token}/{seed}/logs/
```

* `{env}`:
  * SMAC: sc2
  * Multi-agent rendezvous: robot
* `learner`:
  * Algorithm MADDPG: maddpg
  * Algorithm M3DDPG: m3ddpg
  * Algorithm ROM-Q: romq
  * Algorithm ERNIE: ernie
  * Algorithm ROMAX: romax
  * Algorithm MIR3: mir3
* `{token}, {seed}, {step}`: The parameters mentioned before.
* `{timestep}`：Models for different periods. Modify the interval in the algorithm config file.

### Attack the Models and Save the Adversarial Policy

To attack the models, use the following command for example:

```bash
python -u main.py --alg maddpg_traitor_mi_smac --env sc2 --token attack --map 4m_vs_3m --seed 0 --victim_checkpoint ./results/sc2/4m_vs_3m/none/mir3/train/0/models/5000000/
```

* `--alg --env --token --map --seed`: The parameters mentioned before.
* `--victim_checkpoint`: Directory of the model to attack.

Adversarial agent IDs can be modified in the algorithm config file(`./configs/alg/maddpg_traitor_mi_smac.yaml`)

The models and attack datas are saved in the directories like:

```bash
models: ./results/{env}/{map}/traitor_ca_mi/maddpg_ca_mi/{token}/{seed}/models/{timestep}/
datas: ./results/{env}/{map}/traitor_ca_mi/maddpg_ca_mi/{token}/{seed}/logs/
```

* `{env}, {map}, {token}, {seed}, {timestep}`: The parameters mentioned before.

## Demo Videos

We record the behaviors of the agents under the attack in the videos. These videos showcase our methods alongside the baseline methods in the *4m vs 3m* and *9m vs 8m* scenario of the SMAC MARL environment, the *rendezvous* environment and the *real robot* environment, as illustrated in the table below.

| Training algorithm | SMAC 4m vs 3m                              | SMAC 9m vs 8m                              | rendezvous                       | real robot                       |
| ------------------ | ------------------------------------------ | ------------------------------------------ | -------------------------------- | -------------------------------- |
| MADDPG             | ![](video\SMAC\4m_vs_3m_MADDPG\MADDPG.gif) | ![](video\SMAC\9m_vs_8m_MADDPG\MADDPG.gif) | ![](video\rendezvous\MADDPG.gif) | ![](video\real_robot\MADDPG.gif) |
| M3DDPG             | ![](video\SMAC\4m_vs_3m_MADDPG\M3DDPG.gif) | ![](video\SMAC\9m_vs_8m_MADDPG\M3DDPG.gif) | ![](video\rendezvous\M3DDPG.gif) | ![](video\real_robot\M3DDPG.gif) |
| ROM-Q              | ![](video\SMAC\4m_vs_3m_MADDPG\ROMQ.gif)   | ![](video\SMAC\9m_vs_8m_MADDPG\ROM-Q.gif)  | ![](video\rendezvous\ROM-Q.gif)  | ![](video\real_robot\ROM_Q.gif)  |
| ERNIE              | ![](video\SMAC\4m_vs_3m_MADDPG\ERNIE.gif)  | ![](video\SMAC\9m_vs_8m_MADDPG\ERNIE.gif)  | ![](video\rendezvous\ERNIE.gif)  | ![](video\real_robot\ERNIE.gif)  |
| ROMAX              | ![](video\SMAC\4m_vs_3m_MADDPG\ROMAX.gif)  | ![](video\SMAC\9m_vs_8m_MADDPG\ROMAX.gif)  | ![](video\rendezvous\ROMAX.gif)  | ![](video\real_robot\ROMAX.gif)  |
| MIR3               | ![](video\SMAC\4m_vs_3m_MADDPG\MIR3.gif)   | ![](video\SMAC\9m_vs_8m_MADDPG\MIR3.gif)   | ![](video\rendezvous\MIR3.gif)   | ![](video\real_robot\MIR3.gif)   |

