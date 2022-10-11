# BBBC
Belief based Behavior Cloning approach applied to a POMDP in Long-Horizon Tasks.

Install pyTorch and gym requirements (I used an anaconda Python3.6 virtual env).
## Implementation info

 - trainers directory contains separate trainer classes for belief and policy as well as common base class
 - configuration is done using config/learning_params.yaml for network parameters and config/dataset_index.yaml for dataset paths as well as action and observation space dimensionality
 - eval_env.py contains class for evaluating on state-space and mujoco

## Usage

 - tensorboard --logdir ./tensorboard/ to call plots of learning progress and test results

### Training a Model on State-Space Data

1. Create conda environment with conda create --name BBBC --file requirements.txt
2. Generate Data for belief and policy training using https://github.com/alec3010/InvertedPendulum_KalmanLQR (Refer to Repo's README file for instructions, BBBC environment works for data creation as well)
3. Create Data directory in root folder of repo
4. Copy created datasets into data directory
5. Copy relative paths (right-click-> copy relative path) of belief and policy dataset into dataset_index.yaml file
6. python train_belief.py to train belief
7. python train_policy.py to train policy based on previously trained belief and test on state-space
8. python test.py to only run test on trained policy
