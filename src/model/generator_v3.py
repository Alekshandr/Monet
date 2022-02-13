from model.blocks import *

class G(nn.Module):
    def __init__(self, channels=3, conv_dim=64, g_blocks=9):
        
        super(G, self).__init__()

        l = [nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, conv_dim, 7),  #(3, 64) 
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(inplace=True)]

        l += [
            *GBlockUp(conv_dim, conv_dim*2),                #(64, 128)
            *GBlockUp(conv_dim*2, conv_dim*4)                 #(128,  256)
            ]
            
        for _ in range(g_blocks):
             l += [GBlock(conv_dim*4)]
            
        l += [*GBlockDown(conv_dim*4, conv_dim*2),      #(256, 128)
              *GBlockDown(conv_dim*2, conv_dim, padding=0)        #(128, 64)
             ]        #(64, 3)
            
        l += [nn.ReflectionPad2d(channels), 
              nn.Conv2d(conv_dim, channels, 6), #7), 
              nn.Tanh()]
            
        self.model = nn.Sequential(*l)
        
    def forward(self, x):
        return self.model(x)
