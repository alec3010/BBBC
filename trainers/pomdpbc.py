import os
import numpy as np
import torch
#from torch.utils.tensorboard import SummaryWriter

from models import POMDPModel




class POMDPBC():
    def __init__(self, belief_dim, action_dim, obs_dim, actor_hidden_dim, lr) -> None:
        self.agent = POMDPModel(belief_dim=belief_dim, 
                                  obs_dim=obs_dim, 
                                  actor_hidden_dim=actor_hidden_dim, 
                                  action_dim=action_dim).cuda()
        #self.writer = SummaryWriter(log_dir = "./tensorboard")
        self.traj_nr = 0
        self.init_state = torch.cuda.DoubleTensor(belief_dim).fill_(0)
        self.init_ac = torch.cuda.DoubleTensor(action_dim).fill_(0) 
        self.curr_ob = torch.cuda.DoubleTensor(obs_dim).fill_(0)
        self.curr_memory = {
        'curr_ob': curr_ob,    # o_t
        'prev_belief': init_state,   # b_{t-1}
        'prev_ac': init_ac,  # a_{t-1}
        'prev_ob': curr_ob.clone(), # o_{t-1}
        }

        

    def train(self, train, val, batch_size,model_dir="./models", tensorboard_dir="./tensorboard"):
    
    
        print("... train model")
        self.agent.train()

        n_iters = 0
        train_loss, train_cor = 0,0
        for epoch in range(5):

            for traj in train:
                
                for point in traj:

                    self.curr_memory['curr_ob'] = point['obs']
                    targets = point['acs']
                    self.optimizer.zero_grad() # reset weights
                    outputs, belief = self.agent(self.curr_memory) # agent, pytorch
                    self.curr_memory['prev_belief'] = belief.detach()
                    self.curr_memory['prev_ac'] = outputs.detach()
                    self.curr_memory['prev_obs'] = point['obs']
                    
                    loss = self.criterion(outputs, targets) # mse loss
                    loss.backward(retain_graph=True) # backprop
                    self.optimizer.step() # adam optim, gradient updates

                    train_loss+=loss.item()
                    n_iters+=1
                    # self.writer.add_scalar("Loss/train", loss.item(), n_iters)
                
    
                print(f'average train trajectory loss: {(train_loss / n_iters)}')

            
            self.traj_nr += 1
                #print(f'average train batch accuracy: {(train_cor / (batch_size*n_iters))}')
                
        # form data loader for validation (currently predicts on whole valid set)
        # valid_loss, valid_acc = 0,0
        # self.agent.eval()
        # with torch.no_grad():
        #     for i, (valid_inputs,valid_targets) in enumerate(valid_loader):
        #         valid_outputs = self.agent(valid_inputs)
        #         valid_loss += self.criterion(valid_outputs,valid_targets).item()
        #         """ accuracy
        #         _, valid_predicted = torch.max(torch.abs(valid_outputs),1) 
        #         _, valid_targetsbinary = torch.max(torch.abs(valid_targets),1)
        #         valid_correct = (valid_predicted==valid_targetsbinary).sum().item()
        #         valid_acc+=(valid_correct/valid_targets.shape[0])
        #         """
        #     print(f'valid set loss: {valid_loss/len(valid_loader)}')
        #     #print(f'valid set accuracy: {valid_acc}')

        
        torch.save(self.agent.state_dict(), os.path.join(model_dir,"InvertedPendulum.pkl"))
        print("Model saved in file: %s" % model_dir)