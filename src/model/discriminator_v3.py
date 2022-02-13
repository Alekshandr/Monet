from model.blocks import *

class D(nn.Module):

    def __init__(self, channels=3, img_size=299, conv_dim=None):
        
        super(D, self).__init__()
        self.img_size = img_size
        self.channels = channels
        
        self.model = models.inception_v3(pretrained=True)
        num_labels = self.channels * D.out_len() * D.out_len()

        self.model.AuxLogits.fc = nn.Linear(768, num_labels)
        self.model.aux_logits = False
        self.model.fc = nn.Linear(2048, num_labels)

        # Заменяем Fully-Connected слой на наш линейный классификатор
        self.model.classifier = nn.Linear(25088, num_labels)

        self.model = self.model.to(device)
        
    def forward(self, x):
        return self.model(x).view(-1, self.channels, D.out_len(), D.out_len())

    @staticmethod
    def channels():
        return 3
    
    @staticmethod
    def out_len():
        return 299 // 2 ** 5
    
    @staticmethod
    def out_size():
        return D.channels(), D.out_len(), D.out_len() 
