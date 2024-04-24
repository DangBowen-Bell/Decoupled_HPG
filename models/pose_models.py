import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import rotation_tools


#! ######################################################################
#! Pose VAE

# Input: BxR*3
# Output: B*Rx3x3
class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


# Input: BxR*3, BxA 
# Output: BxZ, BxZ
class PoseVAEEncoder(nn.Module):
    def __init__(self, in_dim=21*3, h_dim=512, action_num=4, z_dim=32):
        super(PoseVAEEncoder, self).__init__()
        
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.bn2 = nn.BatchNorm1d(h_dim + action_num)
        self.fc2 = nn.Linear(h_dim + action_num, h_dim)
        self.fc_mu = nn.Linear(h_dim + action_num, z_dim)
        self.fc_logvar = nn.Linear(h_dim + action_num, z_dim)
        
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        
    def forward(self, x, a):
        net = self.bn1(x)
        net = self.fc1(net)
        net = F.leaky_relu(net)
        
        net = torch.cat([net, a], dim=1)
        net = self.bn2(net)
        net = self.fc2(net)
        net = F.leaky_relu(net)
        
        net = torch.cat([net, a], dim=1)
        mu = self.fc_mu(net)
        logvar = self.fc_logvar(net)
        
        return mu, logvar


# Input: BxZ, BxA
# Output: BxR*3
class PoseVAEDecoder(nn.Module):
    def __init__(self, z_dim=32, h_dim=512, action_num=4, out_dim=21*6):
        super(PoseVAEDecoder, self).__init__()
        
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim + action_num, h_dim)
        self.fc_out = nn.Linear(h_dim + action_num, out_dim)
        self.crrd = ContinousRotReprDecoder()
        
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        
    def forward(self, z, a):
        net = self.fc1(z)
        net = F.leaky_relu(net)
        
        net = torch.cat([net, a], dim=1)
        net = self.fc2(net)
        net = F.leaky_relu(net)
        
        net = torch.cat([net, a], dim=1)
        net = self.fc_out(net)

        net = self.crrd(net)
        
        return net

    
# Input: BxR*3, BxA 
# Output: {BxRx3, BxRx9}, BxZ, BxZ
class PoseVAE(nn.Module):
    def __init__(self, options):
        super(PoseVAE, self).__init__()
        if options.use_orient:
            joint_num = 22
        else:
            joint_num = 21
        
        self.encoder = PoseVAEEncoder(in_dim=joint_num*3, 
                                      h_dim=options.h_dim,
                                      action_num=options.action_num, 
                                      z_dim=options.z_dim)
        self.decoder = PoseVAEDecoder(z_dim=options.z_dim, 
                                      h_dim=options.h_dim,
                                      action_num=options.action_num, 
                                      out_dim=joint_num*6)
       
        self.z_dim = options.z_dim
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std    
    
    def generate(self, a):
        batch_size = a.shape[0]
        
        z = torch.randn([batch_size, self.z_dim]).cuda()
        out = self.decoder(z, a)
        
        new_out = {
            'aa': rotation_tools.matrot2aa(out).view(batch_size, -1, 3),
            'rotmat': out.view(batch_size, -1, 9)
        }
        
        return new_out
        
    def forward(self, x, a):
        batch_size = x.shape[0]
        
        x = x.view(batch_size, -1)
        
        mu, logvar = self.encoder(x, a)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z, a)

        new_out = {
            'aa': rotation_tools.matrot2aa(out).view(batch_size, -1, 3),
            'rotmat': out.view(batch_size, -1, 9)
        }
        
        return new_out, mu, logvar
        