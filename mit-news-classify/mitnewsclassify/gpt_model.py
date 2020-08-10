"""
@author: jamiefu
"""
from torch import nn

class GPTModel(nn.Module):
    
    #define all the layers used in model
    def __init__(self, hidden_dim, output_dim):
        
        #Constructor
        super().__init__()          
        
        self.lin1 = nn.Linear(hidden_dim, 512)
        self.drop1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()

        self.lin2 = nn.Linear(512, 256)
        self.drop2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.lin3 = nn.Linear(256, 128)
        self.drop3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        self.lin4 = nn.Linear(128, 256)
        self.drop4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()

        self.lin5 = nn.Linear(256, output_dim)
        self.drop5 = nn.BatchNorm1d(output_dim)
        
        #activation function
        self.act = nn.Sigmoid()
        
    def forward(self, hidden):
        one = self.relu1(self.drop1(self.lin1(hidden)))
        two = self.relu2(self.drop2(self.lin2(one)))
        three = self.relu3(self.drop3(self.lin3(two)))
        four = self.relu4(self.drop4(self.lin4(three)))
        logits = self.drop5(self.lin5(four))

        return logits