import gym
import mujoco_py
import torch

from utils import helpers as h

class EvaluationEnvironment:
    def __init__(self, agent, env_name, obs_idx_list, config) -> None:
        self.config = config
        self.env_name = env_name    
        self.agent = agent
        self.get_params()
        self.rendering = False                      
        self.n_test_episodes = 10
        self.idx_list = obs_idx_list
              
        self.reset_memory()  

        # load agent
        
        
        self.env = gym.make(env_name)

    def eval(self):

        episode_rewards = []
        if self.network_arch == "belief":
            for i in range(self.n_test_episodes):
                episode_reward = self.run_episode_belief()
                episode_rewards.append(episode_reward)
        if self.network_arch == "naive":
            for i in range(self.n_test_episodes):
                episode_reward = self.run_episode_naive()
                episode_rewards.append(episode_reward)

        return sum(episode_rewards)/len(episode_rewards)

    def run_episode_belief(self, max_timesteps=500):
    
        episode_reward = 0
        step = 0

        state = self.env.reset()
        while True:
        
            # get action
            self.agent.eval()
            obs = []
            for idx in self.idx_list:
                obs.append(state[idx])
            self.curr_memory['curr_ob'] = torch.cuda.FloatTensor(obs)    
            
            tensor_action, belief = self.agent(self.curr_memory)
            a = tensor_action.detach().cpu().numpy()[0]

            

            self.curr_memory['prev_belief'] = belief.detach()
            self.curr_memory['prev_ac'] = tensor_action
            self.curr_memory['prev_obs'] = self.curr_memory['curr_ob']

            next_state, r, done, info = self.env.step(a)   
            episode_reward += r       
            state = next_state
            step += 1
            
            if self.rendering:
                env.render()

            if step > max_timesteps: 
                break

        return episode_reward

    def run_episode_naive(self, max_timesteps=500):
        episode_reward = 0
        step = 0
        state = self.env.reset()
        self.agent.eval()
        while True:
            obs = []
            for idx in self.idx_list:
                obs.append(state[idx])

            input = torch.cuda.FloatTensor(obs) 

            tensor_action = self.agent(input)
            a = tensor_action.detach().cpu().numpy()[0]

            next_state, r, done, info = self.env.step(a)   
            episode_reward += r       
            state = next_state
            step += 1

            if self.rendering:
                env.render()

            if step > max_timesteps: 
                break

        return episode_reward

    def get_params(self):
        self.lr = self.config['learning_rate']
        self.process_model = self.config['process_model']
        self.network_arch = self.config['network_arch']
        self.epochs = self.config['epochs']
        self.eval_int = self.config['eval_interval']
        self.shuffle = self.config['shuffle']
        self.batch_size = self.config['batch_size']
        dataset_idx = h.get_params("configs/dataset_index.yaml")
        entry = dataset_idx[self.env_name]
        self.obs_dim = len(entry['obs_dim'][self.process_model])
        self.acs_dim = entry['acs_dim']

    def reset_memory(self):
        self.init_state = torch.cuda.DoubleTensor(self.agent.belief_dim).fill_(0)
        self.init_ac = torch.cuda.DoubleTensor(self.acs_dim).fill_(0) 
        self.curr_ob = torch.cuda.DoubleTensor(self.obs_dim).fill_(0)
        self.curr_memory = {
        'curr_ob': self.curr_ob,    # o_t
        'prev_belief': self.init_state,   # b_{t-1}
        'prev_ac': self.init_ac,  # a_{t-1}
        'prev_ob': self.curr_ob.clone(), # o_{t-1}
        }