from trainers.belief_module_bc import BeliefModuleBC
from trainers.naive_model_bc import NaiveModelBC

import utils.helpers as h

import pickle


def quick_test(env, configs):
    print(env)
    
    if configs['network_arch'] == "naive":
        bc = NaiveModelBC(env, configs=configs)

    if configs['network_arch'] == "belief":
        bc = BeliefModuleBC(env, configs=configs)
        
    bc.train_policy()
    result = bc.get_results()

    return result
    



if __name__ == "__main__":
    results = []
    configs = h.get_params("./configs/learning_params.yaml")
    l_config=configs['learning_params']
    
    for mode in configs['experiment']['prev_acs']:
        l_config['prev_acs'] = mode
        for dim in configs['experiment']['belief_dims']:
            l_config['belief_dim'] = dim
            for length in configs['experiment']['traj_lengths']:
                l_config['traj_length'] = length
                for env in configs['experiment']['envs']:
                    
                    result = quick_test(env, l_config)
                    results.append(result)
                    
    
    configs = h.get_params("./configs/learning_params.yaml")
    configs['network_arch'] = 'naive'
    configs['process_model'] = 'pomdp'
    for length in configs['experiment']['traj_lengths']:
        l_config['traj_length'] = length
        for env in configs['experiment']['envs']:
            
            result = quick_test(env, l_config)
            results.append(result)
    configs['process_model'] = 'mdp'
    for length in configs['experiment']['traj_lengths']:
        l_config['traj_length'] = length
        for env in configs['experiment']['envs']:
            
            result = quick_test(env, l_config)
            results.append(result)


    with open('results/results.pkl', 'wb') as f:
        pickle.dump(resuls, f)    

    

