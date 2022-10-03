from trainers.policy_trainer import PolicyTrainer
from trainers.vae_trainer import VAETrainer

import utils.helpers as h

import pickle

if __name__ == "__main__":
    
    results = []
    configs = h.get_params("./configs/learning_params.yaml")
    env = "InvertedPendulum-StateSpace"
    vae_t = VAETrainer(env, configs=configs)
    vae_t.train()
    pol_t = PolicyTrainer(env, configs=configs)
    pol_t.train()
    
