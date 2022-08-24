import gym
import mujoco_py
import torch

from utils import helpers as h

class EvaluationEnvironment:
    def __init__(self, agent, env_name) -> None:
        self.get_params()
        self.rendering = False                      
        self.n_test_episodes = 10                # number of episodes to test

        # load agent
        self.agent = agent
        
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

        init_state = torch.cuda.DoubleTensor(64).fill_(0)
        init_ac = torch.cuda.DoubleTensor(1).fill_(0) 
        curr_ob = torch.cuda.DoubleTensor(2).fill_(0)

        curr_memory = {
            'curr_ob': curr_ob,    # o_t
            'prev_belief': init_state,   # b_{t-1}
            'prev_ac': init_ac,  # a_{t-1}
            'prev_ob': curr_ob.clone(), # o_{t-1}
            }

        state = self.env.reset()
        while True:
        
            # get action
            self.agent.eval()
            if self.process_model == "pomdp":
                curr_memory['curr_ob'] = torch.from_numpy(state[0:2]).float().cuda()

            if self.process_model == "mdp":
                curr_memory['curr_ob'] = torch.from_numpy(state).float().cuda()
            
            tensor_action, belief = self.agent(curr_memory)
            a = tensor_action.detach().cpu().numpy()[0]

            curr_memory['prev_belief'] = belief.detach()
            curr_memory['prev_ac'] = tensor_action
            curr_memory['prev_obs'] = curr_memory['curr_ob']

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
            if self.process_model == "pomdp":
                input = torch.from_numpy(state[0:2]).float().cuda()

            if self.process_model == "mdp":
                input = torch.from_numpy(state).float().cuda()

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
        self.config = h.get_params("./configs/learning_params.yaml")
        self.lr = self.config['learning_rate']
        self.process_model = self.config['process_model']
        self.network_arch = self.config['network_arch']
        self.epochs = self.config['epochs']
        self.eval_int = self.config['eval_interval']
        self.shuffle = self.config['shuffle']
        self.batch_size = self.config['batch_size']