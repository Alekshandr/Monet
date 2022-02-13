from utils.utils import *

class MonetDataset(Dataset):
    def __init__(self, root='Monet', mode='Train', img_size=None):
        self.imagesA = []
        self.imagesB = []
        
        path = os.path.join(root, f"{mode}A")# + "/*.*"
        
        print(path)
        
        for path, _, files in tqdm(os.walk(path)):
            for file in files:
                try:
                    self.imagesA.append(Image.open(os.path.join(path, file)).copy())
                except:
                     pass

        path = os.path.join(root, f"{mode}B")# + "/*.*"
        print(path)

        for path, _, files in tqdm(os.walk(path)):
            for file in files:
                try:
                    self.imagesB.append(Image.open(os.path.join(path, file)).copy())
                except:
                     pass
                    
        img_height = img_size  
        img_width = img_size 
        self.color_transform = transforms.Compose([
#                     transforms.RandomResizedCrop(img_size),
#                     #transforms.RandomRotation(degrees=(-15,15)),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),   
            
                        transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
                        transforms.RandomCrop((img_height, img_width)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),            
                ])
        
    def __len__(self):
        return max(len(self.imagesA), len(self.imagesB))

    def __getitem__(self, idx):
        imgA = self.imagesA[idx % len(self.imagesA)]
        imgB = self.imagesB[idx % len(self.imagesB)]

        X = self.color_transform(imgA)
        Y = self.color_transform(imgB)
        
        # корректируем нормирование картинки
        X *= 0.5 
        X += 0.5
        Y *= 0.5 
        Y += 0.5
        
        return X, Y
