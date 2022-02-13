import utils.utils
from utils.consts_v3 import *

from model.CycleGan import *
from model.generator_v3 import *
from model.discriminator_v3 import *
from data.dataset import *

def Test(device):
    for b, (A, B) in tqdm(enumerate(loader_test)):

        real_A = A.to(device)
        real_B = B.to(device)

        cycleGAN.g_ab.eval()
        cycleGAN.g_ba.eval()

        fake_A = cycleGAN.g_ba(real_B)
        fake_B = cycleGAN.g_ab(real_A)

        l = real_A.size(0)
        for i in range(l):
            rA = real_A[i].detach().cpu()
            fB = fake_B[i].detach().cpu()
        
            rB = real_B[i].detach().cpu()
            fA = fake_A[i].detach().cpu()

            image_grid = torch.cat((rA, fB, rB, fA), 1)
            save_image(image_grid, "../images/%s/%s-%s.png" % (dataset_name, b, i), normalize=False)

        del A
        del B
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

dataset_test = MonetDataset('../data', mode='Test', img_size=img_size)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

print(len(loader_test.dataset))

cycleGAN = CycleGAN(loader_test, conv_dim, img_size, device, G, D)

# загрузим параметры обучения 16й эпохи
epoch=16
cycleGAN.g_ab.load_state_dict(torch.load("../saved_models/%s/G_AB_%d.pth" % (dataset_name, epoch)))
cycleGAN.g_ba.load_state_dict(torch.load("../saved_models/%s/G_BA_%d.pth" % (dataset_name, epoch)))
cycleGAN.d_a.load_state_dict(torch.load("../saved_models/%s/D_A_%d.pth" % (dataset_name, epoch)))
cycleGAN.d_b.load_state_dict(torch.load("../saved_models/%s/D_B_%d.pth" % (dataset_name, epoch)))

# вывод тестовых данных
Test(device)