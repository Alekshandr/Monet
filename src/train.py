import utils.utils
from utils.consts import *

from model.CycleGan import *
from data.dataset import *
from model.generator import *
from model.discriminator import *

# Моне
dataset = MonetDataset('../data', img_size=img_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# печатаем размер загруженного датасета
print(len(loader.dataset))

# выведем первые batch_size картикок Моне и фоток
for X, Y in loader:
    for i in range(batch_size):
        print_Data_images(X[i], Y[i])
    break

# создание экземпляра Гана и его обучение
cycleGAN = CycleGAN(loader, conv_dim, img_size, device, G, D, epochs=epochs)
cycleGAN.train()

