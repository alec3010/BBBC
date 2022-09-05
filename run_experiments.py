from trainers.belief_module_bc import BeliefModuleBC
from trainers.naive_model_bc import NaiveModelBC
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
    
    # for _ in [False]:
    #     print(_)
    #     l_config['prev_acs'] = _
        
    #     for dim in configs['experiment']['belief_dims']:
    #         l_config['belief_dim'] = dim
    #         for length in configs['experiment']['traj_lengths']:
    #             l_config['traj_length'] = length
    #             for env in configs['experiment']['envs']: 
    #                 result = quick_test(env, l_config)
    #                 results.append(result)
    #                 print("####################################################################################################################################")
                
    
    # configs = h.get_params("./configs/learning_params.yaml")
    # l_config=configs['learning_params']
    # l_config['network_arch'] = 'naive'
    # l_config['process_model'] = 'pomdp'
    # for length in configs['experiment']['traj_lengths']:
    #     l_config['traj_length'] = length
    #     for env in configs['experiment']['envs']:
            
    #         result = quick_test(env, l_config)
    #         results.append(result)
    # configs['process_model'] = 'mdp'
    # for length in configs['experiment']['traj_lengths']:
    #     l_config['traj_length'] = length
    #     for env in configs['experiment']['envs']:
            
    #         result = quick_test(env, l_config)
    #         results.append(result)


    # for i in range(len(results)):
    #     print(results[i]['learning_params'])
    #     print(len(results[i]['train_loss']['epoch']))

    # with open('results/results__.pkl', 'wb') as f:
    #     pickle.dump(results, f)    

    

