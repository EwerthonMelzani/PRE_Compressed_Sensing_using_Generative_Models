import torch
import os
from DCGAN import Generator,Discriminator
from torchvision.utils import save_image


Z_DIM = 100
CHANNELS_IMG = 3
FEATURES_GEN = 64
FEATURES_DISC = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device, memory_format=torch.channels_last)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device, memory_format=torch.channels_last)


opt_gen = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_disc = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))


checkpoint_path = os.path.join("Save_Model", "dcgan_final.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)


gen.load_state_dict(checkpoint["gen_state_dict"])
disc.load_state_dict(checkpoint["disc_state_dict"])
opt_gen.load_state_dict(checkpoint["opt_gen_state_dict"])
opt_disc.load_state_dict(checkpoint["opt_disc_state_dict"])

print(f"Checkpoint carregado do epoch {checkpoint['epoch']}")


gen.eval()


with torch.inference_mode():
    noise = torch.randn(32, Z_DIM, 1, 1).to(device)
    fake_images = gen(noise)
    fake_images = fake_images * 0.5 + 0.5  
    save_image(fake_images, "gerado_do_modelo_salvo.png", nrow=8)
    print("Imagem gerada salva como 'gerado_do_modelo_salvo.png'")
