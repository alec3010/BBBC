from trainers.behaviorcloner import BehaviorCloner

import utils.helpers as h

import pickle


def quick_test(env, configs):
    print(env)
    bc = BehaviorCloner(env, configs=configs)
    bc.train()
    result = bc.get_results()
    

    return result
    


if __name__ == "__main__":
    
    results = []
    configs = h.get_params("./configs/learning_params.yaml")
    env = "InvertedPendulum-v2"
    quick_test(env, configs)
    
