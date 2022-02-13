from utils.utils import *

def GBlockUp(channels_in, channels_out, k_size=3, stride=2, padding=1, batchNorm=True):
    l = [nn.Conv2d(channels_in, channels_out, k_size, stride, padding, bias=False)]
    if batchNorm:
        l += [nn.InstanceNorm2d(channels_out)]# [nn.BatchNorm2d(channels_out)]
    l += [#nn.MaxPool2d((stride, stride), stride=(1, 1)),
          nn.ReLU(inplace=True)]
    return l

def GBlockDown(channels_in, channels_out, k_size=3, stride=1, padding=1, batchNorm=True, relu=True):
    l = [nn.Upsample(scale_factor=2),
         nn.Conv2d(channels_in, channels_out, k_size, stride=stride, padding=padding)
         #nn.ConvTranspose2d(channels_in, channels_out, k_size, stride=stride, padding=padding, bias=False)
        ]
    if batchNorm:
        l += [nn.InstanceNorm2d(channels_out)] # [nn.BatchNorm2d(channels_out)] #nn.InstanceNorm2d(
    if relu:
        #l += [nn.LeakyReLU(0.2, inplace=True)]
        l += [nn.ReLU(inplace=True)]
    
    return l

class GBlock(nn.Module):
    def __init__(self, channels_in):
        super(GBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels_in, channels_in, 3),
            nn.InstanceNorm2d(channels_in),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels_in, channels_in, 3),
            nn.InstanceNorm2d(channels_in),
        )

    def forward(self, x):
        #return torch.cat((x, self.block(x)), 1)
        return x + self.block(x)
        #return self.block(x)
    
def DBlock(channels_in, channels_out, k_size=3, stride=2, padding=1, normalize=True):
    return [        
        nn.Conv2d(channels_in, channels_out, k_size, stride, padding),
        nn.LeakyReLU(),
        nn.Conv2d(channels_out, 3, (3, 3), padding=1),
        nn.Sigmoid()
    ]
    
def DBlockLRelu(channels_in, channels_out, k_size=4, stride=2, padding=1, normalize=True):
    l = [nn.Conv2d(channels_in, channels_out, k_size, stride=stride, padding=padding)]
    if normalize:
        l += [nn.InstanceNorm2d(channels_out)]
    l += [nn.LeakyReLU(0.2, inplace=True)]
    #l += [nn.ReLU()]
    return l

def DBlockSigmoid(channels_in, channels_out, k_size=3, stride=2, padding=1, normalize=True):
    l = [nn.Conv2d(channels_in, channels_out, k_size, stride=stride, padding=padding)]
    if normalize:
        l += [nn.InstanceNorm2d(channels_out)]
    l += [nn.Sigmoid()]
    return l

