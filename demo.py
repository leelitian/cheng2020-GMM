import torch
from PIL import Image
from torchvision import transforms
from model import Cheng2020GMM

torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'  # use cpu here will be faster
    checkpoint = torch.load('checkpoint.pth.tar', map_location=device)

    net = Cheng2020GMM().to(device).eval()
    net.load_state_dict(checkpoint["state_dict"])

    img = Image.open('./images/kodim01.png').convert('RGB')
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # codec
        out = net.compress(x, 'foo')
        rec = net.decompress('foo')
        rec = transforms.ToPILImage()(rec['x_hat'].squeeze().cpu())
        rec.save('./images/codec.png', format="PNG")

        # inference
        out = net(x)
        rec = out['x_hat'].clamp(0, 1)
        rec = transforms.ToPILImage()(rec.squeeze().cpu())
        rec.save('./images/infer.png', format="PNG")

        print('saved in ./images')