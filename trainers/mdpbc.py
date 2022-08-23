import torch
#from torch.utils.tensorboard import SummaryWriter

from models import MDPModel
import os
import matplotlib as mpl
import matplotlib.pyplot as plt




class MDPBC():
    def __init__(self, lr) -> None:
        self.agent = MDPModel(belief_dim=4, obs_dim=4, actor_hidden_dim=64, action_dim=1).cuda()
        self.optimizer = torch.optim.Adam(self.agent.parameters(),lr=lr)# adam optimization
        self.criterion = torch.nn.MSELoss()# MSE loss
        self.writer = SummaryWriter(log_dir = "./tensorboard")
        self.traj_nr = 0
        

    def train(self, train_x, train_y, val_x, val_y, batch_size,model_dir="./models", tensorboard_dir="./tensorboard"):
         
        
    
        print("... train model")

        
        
        # for traj in train:
        # form data loader for training
        train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
        self.agent.train()

        n_iters = 0
        train_loss, train_cor = 0,0
        for i in range(2):
            train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,shuffle=True)
            
            for batch_idx, (inputs,targets) in enumerate(train_loader):

                self.optimizer.zero_grad() # reset weights
                outputs = self.agent(inputs) # agent, pytorch
                loss = self.criterion(outputs,targets) # mse loss
                loss.backward() # backprop
                self.optimizer.step() # adam optim, gradient updates

                train_loss+=loss.item()
                n_iters+=1
                self.writer.add_scalar("Loss/train", loss.item(), n_iters)
                
    
            print(f'average train batch loss: {(train_loss / n_iters)}')

            
            self.traj_nr += 1
                #print(f'average train batch accuracy: {(train_cor / (batch_size*n_iters))}')
                
            # form data loader for validation (currently predicts on whole valid set)
            # for demo in val:
            #     valid_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(demo["obs"]),torch.FloatTensor(demo["acs"]))
            #     valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=len(demo["obs"]),shuffle=False)
            #     valid_loss, valid_acc = 0,0
            #     self.agent.eval()
            #     with torch.no_grad():
            #         for i, (valid_inputs,valid_targets) in enumerate(valid_loader):
            #             valid_inputs = valid_inputs.unsqueeze(1).float()
            #             valid_outputs = self.agent(valid_inputs)
            #             valid_loss += self.criterion(valid_outputs,valid_targets).item()
            #             """ accuracy
            #             _, valid_predicted = torch.max(torch.abs(valid_outputs),1) 
            #             _, valid_targetsbinary = torch.max(torch.abs(valid_targets),1)
            #             valid_correct = (valid_predicted==valid_targetsbinary).sum().item()
            #             valid_acc+=(valid_correct/valid_targets.shape[0])
            #             """
            #         print(f'valid set loss: {valid_loss}')
            #         #print(f'valid set accuracy: {valid_acc}')

        
        torch.save(self.agent.state_dict(), os.path.join(model_dir,"InvertedPendulum.pkl"))
        print("Model saved in file: %s" % model_dir)
