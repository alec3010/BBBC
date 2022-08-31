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
    
    for _ in [True]:
        print(_)
        l_config['prev_acs'] = _
        
        for dim in configs['experiment']['belief_dims']:
            l_config['belief_dim'] = dim
            for length in configs['experiment']['traj_lengths']:
                l_config['traj_length'] = length
                for env in configs['experiment']['envs']: 
                    print("belief dim in config: ", l_config['prev_acs'])
                    result = quick_test(env, l_config)
                    print("belief dim in result: ", result['learning_params']['prev_acs'])
                    results.append(result)
                    print("belief dim in lass list item: ", results[-1]['learning_params']['prev_acs'])
                    print("####################################################################################################################################")
                
    
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


    for i in range(len(results)):
        print(results[i]['learning_params'])
        print(len(results[i]['train_loss']['epoch']))

    with open('results/results__acs.pkl', 'wb') as f:
        pickle.dump(results, f)    

    

