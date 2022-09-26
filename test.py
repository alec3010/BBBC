import torch
from trainers.policy_trainer import PolicyTrainer
from eval_env import EvaluationEnvironment


if __name__ == "__main__":
    
    results = []
    configs = h.get_params("./configs/learning_params.yaml")
    env = "InvertedPendulum-v2"
    pol_t = PolicyTrainer(env, configs=configs)
    pol_t.model.load_state_dict(torch.load(configs['policy_state_dict']))
    pol_t.eval_on_ss()