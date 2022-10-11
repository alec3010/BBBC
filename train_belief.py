from trainers.policy_trainer import PolicyTrainer
from trainers.ae_trainer import AETrainer

import utils.helpers as h

import pickle

if __name__ == "__main__":
    
    results = []
    configs = h.get_params("./configs/learning_params.yaml")
    env = "InvertedPendulum-StateSpace"
    vae_t = AETrainer(env, configs=configs)
    vae_t.train()

