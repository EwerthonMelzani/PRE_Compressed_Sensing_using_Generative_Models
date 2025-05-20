
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torchvision.utils import save_image
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader


torch.manual_seed(42)

class Discriminator(nn.Module):
  def __init__(self, channels_img,features_d):
    super(Discriminator, self).__init__()
    self.discriminator = nn.Sequential(
        # Input: N x channels_img x 64 x 64
        nn.Conv2d(
          channels_img, features_d, kernel_size=4, stride=2, padding=1
        ),
        nn.LeakyReLU(0.2),
        self._block(features_d, features_d*2, 4, 2, 1), # 16 x 16
        self._block(features_d*2, features_d*4, 4, 2, 1), # 8 x 8
        self._block(features_d*4, features_d*8, 4, 2, 1), # 4 x 4
        nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0), # 1 x 1
        nn.Sigmoid()
    )

  def _block(self,in_channels, out_channels , kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2),
    )

  def forward(self,x):
    return self.discriminator(x)

class Generator(nn.Module):
  def __init__(self, z_dim, channels_img, features_g):
    super(Generator, self).__init__()
    self.net = nn.Sequential(
        # Input : N x z_dim x 1 x 1
        self._block(z_dim, features_g*16, 4, 1, 0), # N x f_g*16 x 4 x 4
        self._block(features_g*16, features_g*8, 4, 2, 1), # 8 x 8
        self._block(features_g*8, features_g*4, 4, 2, 1), # 16 x 16
        self._block(features_g*4, features_g*2, 4, 2, 1), # 32 x 32
        nn.ConvTranspose2d(
            features_g*2, channels_img, kernel_size=4, stride=2, padding=1 # 64 x 64
        ),
        nn.Tanh(), # [-1, 1]
    )


  def _block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

  def forward(self,x):
    return self.net(x)


def initialize_weight(model):
  for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
      nn.init.normal_(m.weight.data, 0.0, 0.02)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 30 #5
FEATURES_DICS = 64
FEATURES_GEN = 64

transform_CelebA = T.Compose(
    [
        T.Resize(IMAGE_SIZE),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)] )

    ]
)


dataset = datasets.CelebA(root="dataset",transform = transform_CelebA, split="all", download=False)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device,memory_format=torch.channels_last)
disc = Discriminator(CHANNELS_IMG, FEATURES_DICS).to(device,memory_format=torch.channels_last)

initialize_weight(gen)
initialize_weight(disc)


opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)



SAVE_DIR = "Save_Model"
os.makedirs(SAVE_DIR, exist_ok=True)

if __name__ == "__main__":


    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device,memory_format=torch.channels_last)
            noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device,memory_format=torch.channels_last)

            gen.train()
            disc.train()

            ### Train Discriminator
            for _ in range(1):
                disc_real = disc(real).reshape(-1)
                loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))


                disc_fake = disc(gen(noise)).reshape(-1)
                loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))


                loss_disc = (loss_disc_real + loss_disc_fake) / 2


                disc.zero_grad(set_to_none=True)

                loss_disc.backward()

                opt_disc.step()

            ### Train Generator min log(1-D(G(z))) <--> max log(D(G(z)))
            for _ in range(2):

                output = disc(gen(noise)).reshape(-1)
                loss_gen = criterion(output, torch.ones_like(output))


                gen.zero_grad()


                loss_gen.backward()


                opt_gen.step()


            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )   

            ### Evaluation Creating a image

            gen.eval()
            disc.eval()

            with torch.inference_mode():
                fake_imgs = gen(fixed_noise)
                fake_imgs = fake_imgs * 0.5 + 0.5
                save_image(fake_imgs, f"{SAVE_DIR}/fake_epoch_{epoch+1}.png", nrow=8)



    torch.save({
        'gen_state_dict': gen.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'opt_gen_state_dict': opt_gen.state_dict(),
        'opt_disc_state_dict': opt_disc.state_dict(),
        'epoch': NUM_EPOCHS,
    }, os.path.join(SAVE_DIR, "dcgan_final.pth"))














