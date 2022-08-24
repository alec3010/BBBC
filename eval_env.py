import gym
import mujoco_py
import torch



class EvaluationEnvironment:
    def __init__(self, agent, env_name) -> None:

        self.rendering = False                      
        self.n_test_episodes = 10                # number of episodes to test

        # load agent
        self.agent = agent
        
        self.env = gym.make(env_name)

        self.episode_rewards = []

    def evaluate(self):

        for i in range(n_test_episodes):
            episode_reward = run_episode(env, agent ,rendering=rendering)
            episode_rewards.append(episode_reward)

    def run_episode(env, agent, rendering=True, max_timesteps=500):
    
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

        state = env.reset()
        while True:
        
            # get action
            agent.eval()
            curr_memory['curr_ob'] = torch.from_numpy(state[0:2]).float().cuda()
            
            tensor_action, belief = agent(curr_memory)
            a = tensor_action.detach().cpu().numpy()[0]

            curr_memory['prev_belief'] = belief.detach()
            curr_memory['prev_ac'] = tensor_action
            curr_memory['prev_obs'] = curr_memory['curr_ob']

            next_state, r, done, info = env.step(a)   
            episode_reward += r       
            state = next_state
            step += 1
            
            if rendering:
                env.render()

            if step > max_timesteps: 
                break

        return episode_reward
