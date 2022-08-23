from trainers.belief_module_bc import BeliefModuleBC
from trainers.naive_model_bc import NaiveModelBC

import utils.helpers as h

if __name__ == "__main__":

    # read data    
    
    # mdpbc = MDPBehaviorCloner(acs_dim=1, obs_dim=2)

    # mdpbc.train_policy()

    pomdpbc = BeliefModuleBC("InvertedPendulum-v2")

    pomdpbc.train_policy()
