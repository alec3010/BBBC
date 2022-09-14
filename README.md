# BBBC
Belief based Behavior Cloning approach applied to a POMDP in Long-Horizon Tasks.

```
|-- model.py             (vae in './models/')
|-- train_vae.py       (train vae)
|-- test_vae.py        (vae performance, stores results in './results/') 
```
Install pyTorch and gym requirements (I used an anaconda Python3.6 virtual env).

Ablations:

 - With/wo belief state
 - With/wo previous actions
 - Different sizes of hidden state in belief module
 - vaes

