from utils.utils import *

class CycleGAN(object):
    lr = 0.00012
    beta1 = 0.5
    beta2 = 0.999
    step_size = 200
    gamma = 0.5
    
    alpha_A = 1.0
    alpha_B = 1.0
    
    lambdaCycle = 5.0
    lambdaIdt = 10.0
    
    epoch_interval = 4
    checkpoint_interval = 4
    
    lossesGAN   = Losses()
    lossesCycle = Losses()
    lossesG   = Losses()
    lossesD = Losses()
    
    def __init__(self, loader, conv_dim, img_size, device, G, D, epochs=16):
        self.loader = loader
        self.device = device
        self.epochs = epochs

        self.D = D
        
        self.g_ab = G(conv_dim=conv_dim)
        self.g_ba = G(conv_dim=conv_dim)
        self.d_a  = D(conv_dim=conv_dim, img_size=img_size)
        self.d_b  = D(conv_dim=conv_dim, img_size=img_size)
        
        self.g_ab = self.g_ab.to(device)
        self.g_ba = self.g_ba.to(device)
        self.d_a  = self.d_a.to(device)
        self.d_b  = self.d_b.to(device)
    
        self.criterionGAN   = nn.MSELoss().to(device) #nn.BCEWithLogitsLoss().to(self.device)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt   = torch.nn.L1Loss()
            
        g_params = itertools.chain(self.g_ab.parameters(), self.g_ba.parameters())
        d_params = itertools.chain(self.d_a.parameters(), self.d_b.parameters())
        
        self.g_optimizer   = torch.optim.Adam(g_params, self.lr, [self.beta1, self.beta2])
        self.d_a_optimizer = torch.optim.Adam(self.d_a.parameters(), self.lr, [self.beta1, self.beta2])
        self.d_b_optimizer = torch.optim.Adam(self.d_b.parameters(), self.lr, [self.beta1, self.beta2])

        self.g_scheduler   = torch.optim.lr_scheduler.StepLR(self.g_optimizer, step_size=self.step_size, gamma=self.gamma)
        self.d_a_scheduler = torch.optim.lr_scheduler.StepLR(self.d_a_optimizer, step_size=self.step_size, gamma=self.gamma)
        self.d_b_scheduler = torch.optim.lr_scheduler.StepLR(self.d_b_optimizer, step_size=self.step_size, gamma=self.gamma)
        
    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_a_optimizer.zero_grad()
        self.d_b_optimizer.zero_grad()

    def train(self):
        prev_time = time.time()
        for epoch in range(self.epochs+1):
            for gray_images, color_images in tqdm(self.loader):
                
                real_A = gray_images.to(self.device)
                real_B = color_images.to(self.device)
                
                valid_images = torch.tensor(np.ones((real_A.size(0), *self.D.out_size())).astype(np.float32), device=self.device, requires_grad=False)
                fake_images  = torch.tensor(np.zeros((real_A.size(0), *self.D.out_size())).astype(np.float32), device=self.device, requires_grad=False)
        
                #  Учим гнераторы
                self.g_ab.train()
                self.g_ba.train()

                self.g_optimizer.zero_grad()

                # считаем лосс идентичности
                lossIdt_A = self.criterionIdt(self.g_ba(real_A), real_A)
                lossIdt_B = self.criterionIdt(self.g_ab(real_B), real_B)

                loss_identity = (lossIdt_A * self.alpha_A + lossIdt_B * self.alpha_B) / 2

                # лосс гана
                fake_B = self.g_ab(real_A)
                
                lossGAN_AB = self.criterionGAN(self.d_b(fake_B), valid_images)
                fake_A = self.g_ba(real_B)
                lossGAN_BA = self.criterionGAN(self.d_a(fake_A), valid_images)

                lossGAN = (lossGAN_AB + lossGAN_BA) / 2

                # циклический лосс
                recov_A = self.g_ba(fake_B)
                lossCycle_A = self.criterionCycle(recov_A, real_A)
                recov_B = self.g_ab(fake_A)
                lossCycle_B = self.criterionCycle(recov_B, real_B)
                lossCycle = (lossCycle_A * self.alpha_A + lossCycle_B * self.alpha_B) / 2

                # Итоговый лосс
                loss_G = self.lambdaCycle * lossCycle + self.lambdaIdt * loss_identity + lossGAN
                
                loss_G.backward()
                self.g_optimizer.step()

                #  учим дискриминатор A
                self.d_a_optimizer.zero_grad()

                # реальный и фейковый лосс A
                loss_real = self.criterionGAN(self.d_a(real_A), valid_images)
                loss_fake = self.criterionGAN(self.d_a(fake_A.detach()), fake_images)
                
                # суммарный лосс
                loss_D_A = (loss_real + loss_fake) / 2

                loss_D_A.backward()
                self.d_a_optimizer.step()

                #  учим дискриминатор B
                self.d_b_optimizer.zero_grad()

                # реальный и фейковый лосс B
                loss_real = self.criterionGAN(self.d_b(real_B), valid_images)
                loss_fake = self.criterionGAN(self.d_b(fake_B.detach()), fake_images)

                # суммарный лосс
                loss_D_B = (loss_real + loss_fake) / 2

                loss_D_B.backward()
                self.d_b_optimizer.step()

                loss_D = (loss_D_A + loss_D_B) / 2
                
                self.lossesGAN.update(lossGAN.cpu().item(),     gray_images.size(0))
                self.lossesCycle.update(lossCycle.cpu().item(), gray_images.size(0))
                self.lossesG.update(loss_G.cpu().item(), gray_images.size(0))
                self.lossesD.update(loss_D.cpu().item(), gray_images.size(0))
                
                # сохраняем первые элементы батчей для промежуточного выводы
                if epoch % self.epoch_interval == 0:
                    xx = real_A[0].detach().cpu()
                    yy = real_B[0].detach().cpu()
                    zz1 = fake_A[0].detach().cpu()#.view(img_size, img_size, -1)
                    zz2 = fake_B[0].detach().cpu()#.view(img_size, img_size, -1)
                
                # очищаем память карты
                del gray_images
                del color_images
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()                        

            # изменяем шаги
            self.g_scheduler.step()
            self.d_a_scheduler.step()
            self.d_b_scheduler.step()

                
            if epoch % self.epoch_interval == 0:
                # Логируем как в примере, на который дали ссылку на данные в slack 
                batches_done = epoch * len(self.loader) + i
                batches_left = self.epochs * len(self.loader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write(\
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s" %
                (
                    epoch,
                    self.epochs,
                    i,
                    len(self.loader.dataset),
                    loss_D.item(),
                    loss_G.item(),
                    lossGAN.item(),
                    lossCycle.item(),
                    loss_identity.item(),
                    time_left,
                ))

                # печать лоссов и выборочных картинок
                print_images(xx, yy, zz1, zz2)
                self.lossesG.plot('loss G')
                self.lossesD.plot('loss D')

            if self.checkpoint_interval != -1 and epoch % self.epoch_interval == 0:
                torch.save(self.g_ab.state_dict(), "../saved_models/%s/G_AB_%d.pth" % (dataset_name, epoch))
                torch.save(self.g_ba.state_dict(), "../saved_models/%s/G_BA_%d.pth" % (dataset_name, epoch))
                torch.save(self.d_a.state_dict(), "../saved_models/%s/D_A_%d.pth" % (dataset_name, epoch))
                torch.save(self.d_b.state_dict(), "../saved_models/%s/D_B_%d.pth" % (dataset_name, epoch))
