import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora
import numpy as np
import random


# Define your network class
class YourNetwork(nn.Module):
    def __init__(self):
        super(YourNetwork, self).__init__()
        self.l1 = nn.Linear(20, 20)
        self.layer1 = nn.Linear(20, 1)  # Example layer

    def forward(self, x):
        x_l1 = self.l1(x)
        x = F.relu(x_l1)
        x = self.layer1(x)
        return x


class YourNetwork1(nn.Module):
    def __init__(self):
        super(YourNetwork1, self).__init__()
        self.l1 = nn.Linear(20, 20)
        self.layer1 = nn.Linear(20, 1)  # Example layer
        self.lora = lora.Linear(20, 20, r=5)

    def forward(self, x):
        x_l1 = self.l1(x)
        x_lora = self.lora(x)
        x = F.relu(x_l1 + x_lora)
        x = self.layer1(x)
        return x


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


set_random_seed(1)
# Create an instance of your network
net = YourNetwork1()

net.load_state_dict(torch.load("/home/ruiqi/projects/meta_adapt/adaptation/test/network.pth"), strict=False)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-1)
lora.mark_only_lora_as_trainable(net, bias='lora_only')
print(net.state_dict()['lora.lora_A'][0, :10])
# load meta params and transform to trainable
meta_params = lora.lora_state_dict(net, bias='lora_only')
meta_params['lora.lora_A'] = 2 * meta_params['lora.lora_A']
# for k in meta_params.keys():
#     meta_params[k] = nn.Parameter(meta_params[k])
print(meta_params['lora.lora_A'][0, :10])

# # replace with the meta params
lora_bias_names = []
# load the loraA and loraB matrix
for name, param in net.named_parameters():
    if 'lora_' in name:
        param.data = meta_params[name]
        param.requires_grad = True
        bias_name = name.split('lora_')[0] + 'bias'
        lora_bias_names.append(bias_name)

# load the lora bias
for name, param in net.named_parameters():
    if name in lora_bias_names:
        param.data = meta_params[name]
        param.requires_grad = True

print(net.state_dict()['lora.lora_A'][0, :10])
for i in range(10):
    x = torch.arange(20).reshape(1, 20).to(torch.float)
    y = torch.tensor([0.1]).reshape(1, 1).to(torch.float)
    pred = net(x)
    loss = F.mse_loss(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(meta_params['lora.lora_A'][0, :10])
print(net.state_dict()['lora.lora_A'][0, :10])
# torch.save(lora.lora_state_dict(net, bias='lora_only'), './test/meta_params.pth')
