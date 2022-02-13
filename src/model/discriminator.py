from model.blocks import *

class D(nn.Module):
    def __init__(self, channels=3, img_size=256, conv_dim=64):
        
        super(D, self).__init__()
        self.img_size = img_size
        self.model = nn.Sequential(
            *DBlockLRelu(channels,   conv_dim, normalize=False), #(3, 64)
            *DBlockLRelu(conv_dim,   conv_dim*2),   #(64, 128),
            *DBlockLRelu(conv_dim*2, conv_dim*4),   #(128, 256),
            *DBlockLRelu(conv_dim*4, conv_dim*8),   #(256, 512),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(conv_dim*8, 1, 4, padding=1)   #(512, 1)
        )
        
    def forward(self, x):
        return self.model(x)

    @staticmethod
    def out_size():
        return 1, 256 // 2 ** 4, 256 // 2 ** 4 
