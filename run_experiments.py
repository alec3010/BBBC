from trainers.belief_module_bc import BeliefModuleBC
from trainers.naive_model_bc import NaiveModelBC

import utils.helpers as h

if __name__ == "__main__":



    # read data    
    
    # mdpbc = MDPBehaviorCloner(acs_dim=1, obs_dim=2)

    # mdpbc.train_policy()
    configs = h.get_params("./configs/learning_params.yaml")
    if configs['network_arch'] == "naive":

        bc = NaiveModelBC("InvertedPendulum-v2")

    if configs['network_arch'] == "belief":
        
        bc = BeliefModuleBC("InvertedPendulum-v2", 10)

    bc.train_policy()
