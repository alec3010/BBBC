from trainers.behaviorcloner import BehaviorCloner

import utils.helpers as h

import pickle


def quick_test(env, configs):
    print(env)
    bc = BehaviorCloner(env, configs=configs)
    bc.train_policy()
    result = bc.get_results()
    

    return result
    


if __name__ == "__main__":
    
    results = []
    configs = h.get_params("./configs/learning_params.yaml")
    l_config=configs['learning_params']
    env = "InvertedPendulum-v2"
    quick_test(env, l_config)
    
