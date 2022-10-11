# BBBC
Belief based Behavior Cloning approach applied to a POMDP in Long-Horizon Tasks.

```
|-- model.py             (vae in './models/')
|-- train_vae.py       (train vae)
|-- test_vae.py        (vae performance, stores results in './results/') 
```
Install pyTorch and gym requirements (I used an anaconda Python3.6 virtual env).


## Usage

### Training a Model on State-Space Data

1. Generate Data using https://github.com/alec3010/InvertedPendulum_KalmanLQR(Refer to Repo's README file for instructions)
2. Create Data directory in root folder
3. Copy created datasets into data directory
4. Copy relative paths (right-click-> copy relative path) of belief and policy Dataset into dataset_index.yaml file
5. python train_belief.py to train belief
6. python train_policy.py to train policy based on previously trained belief
7. python test.py to only run test on trained policy
