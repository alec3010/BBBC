import gym
import mujoco_py
import torch
import numpy as np
from scipy import signal
import math
import matplotlib.pyplot as plt

from utils import helpers as h

class EvaluationEnvironment:
    def __init__(self, agent, env_name, obs_idx_list, config) -> None:
        self.config = config
        
        self.belief_dim = config['belief_dim']
        self.env_name = env_name    
        self.agent = agent
        self.get_params()
        self.rendering = False                      
        self.n_test_episodes = 10
        self.idx_list = obs_idx_list
        self.hidden = None #torch.cuda.DoubleTensor(1,self.belief_dim).fill_(0)
              
        self.reset_memory()  

        # load agent
        
        
        self.env = gym.make(env_name)

    def eval_mjc(self):

        episode_rewards = []
        for i in range(self.n_test_episodes):
            episode_reward = self.run_episode_mjc()
            episode_rewards.append(episode_reward)

        return sum(episode_rewards)/len(episode_rewards)

    def eval_ss(self):
        t_final      = 100

        dt_plant     = 0.1
        dt_control   = 0.1

        t_plant   = np.arange(0, t_final, dt_plant)
        t_control = np.arange(0, t_final, dt_control)

        M_Cart  = 0.5
        M_Arm   = 0.2
        length  = 0.3
        b       = 0.1
        g       = 9.8
        I       = (1/3)*M_Arm*(length**2)

        P = ( I*(M_Cart+M_Arm) + (M_Cart*M_Arm*length**2) )

        A32 = ( (M_Arm**2)*g*(length**2) ) / P
        A33 = ( -(I+M_Arm*(length**2))*b ) /  P
        A42 = ( M_Arm*g*length*(M_Cart+M_Arm) ) / P
        A43 = ( -(M_Arm*length*b) ) / P

        B3 = ( I + M_Arm*(length**2) ) / P
        B4 = ( M_Arm*length ) / P

        A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, A32, A33, 0],
              [0, A42, A43, 0]])

        B = np.array([[0], [0], [B3], [B4]])
        
        C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

        D = np.array([[0],[0]])

        inverse_pendulum_plant = signal.StateSpace(A, B, C, D)

        inverse_pendulum_plant_d = inverse_pendulum_plant.to_discrete(dt_plant)
        A_discrete = inverse_pendulum_plant_d.A
        B_discrete = inverse_pendulum_plant_d.B




        x         = float(0.0)
        theta     = float(0.0)
        x_dot     = float(0.0)
        theta_dot = float(1.0)
        states = np.array([[x], [theta],[x_dot], [theta_dot]])
        states_l = np.copy(states)

        full_step = int(dt_control/dt_plant)
        steps = math.ceil(t_final/dt_control)

        states_coll   = [[],[],[],[]]  # real states
        states_coll_n = [[],[],[],[]]  # states w/ noise 
        control_force_coll = []
        self.agent.eval_mode()    

        print('starting eval loop')   

        for i in range(0, steps):
        
            measurement = h.add_noise(states_l)
            
            z = np.array([ [states_l[0,0]], [states_l[1,0]] ])

            # here agent:
            with torch.no_grad():
                assert np.isnan(z).any() == False
                assert np.isinf(z).any() == False
                
                input = torch.cuda.FloatTensor(z).squeeze(1)
                assert torch.isnan(input).any()==False 
                if torch.isinf(input).any():
                    print(input)
                    print(z)
                assert torch.isinf(input).any()==False
                
                tensor_action = self.agent(input)
                assert torch.isnan(tensor_action).any()==False and torch.isinf(tensor_action).any()==False
                control_force = tensor_action.detach().cpu().numpy()[0]
                print(control_force)

            control_force_coll = np.append(control_force_coll, control_force)
            
            states_coll_n = np.append(states_coll_n, measurement,axis=1)
             # Update states with ss eqs
            assert np.isnan(states_l).any()==False
            assert np.isnan(A).any()==False
            assert np.isnan(B_discrete).any()==False
            assert np.isnan(control_force)==False
            states = np.matmul(A_discrete, states_l) + B_discrete*control_force
            assert np.isnan(states).any()==False
            assert np.max(states) < 10
            if np.min(states) < -10:
                print(i)
                print(states)
                print(control_force)
            assert np.min(states) > -10
            
            
            # Collect variables to plot
            states_coll = np.append(states_coll, states_l,axis=1)
            
            # Store info for next iteration
            states_l = states
            

               

        fig, axs = plt.subplots(2)

        axs[0].plot(t_plant, states_coll[:][0], label='x')
        axs[0].plot(t_plant, states_coll[:][1], label='theta')
        axs[0].plot(t_plant, states_coll[:][2], label='x_dot')
        axs[0].plot(t_plant, states_coll[:][3], label='theta_dot')

        axs[0].legend(loc='best', shadow=True, framealpha=1)

        axs[1].plot(t_control, control_force_coll, label = 'Control Force')
        axs[1].plot(t_plant,  -states_coll[:][2],  label = 'Error')

        axs[1].legend(loc='best', shadow=True, framealpha=1)

        plt.show()



    def run_episode_mjc(self, max_timesteps=500):
    
        episode_reward = 0
        step = 0

        state = self.env.reset()
        #self.agent.reset_memory()
        self.agent.eval()
        self.hidden = None
        while True:
        
            # get action
            
            obs = []
            for idx in self.idx_list:
                obs.append(state[idx])
            input_ = torch.cuda.FloatTensor(obs).unsqueeze(0)   

            if self.network_arch == "RNNFF":
        
                tensor_action, self.hidden = self.agent(input_, self.hidden) # agent, pytorch
            elif self.network_arch == "FF":
                tensor_action = self.agent(input_)

            
            a = tensor_action.detach().cpu().numpy()[0]


            next_state, r, done, info = self.env.step(a)   
            episode_reward += r       
            state = next_state
            step += 1
            
            self.env.render()

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