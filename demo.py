import torch
from PIL import Image
from torchvision import transforms
from model import SwinHyperprior

torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    checkpoint = torch.load('checkpoint.pth.tar', map_location=device)

    net = Hyperprior().to(device).eval()
    net.load_state_dict(checkpoint["state_dict"])

    img = Image.open('./images/kodim01.png').convert('RGB')
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # codec
        out = net.compress(x)
        rec = net.decompress(out['strings'], out['shape'])
        rec = transforms.ToPILImage()(rec['x_hat'].squeeze().cpu())
        rec.save('./images/codec.png', format="PNG")

        # inference
        out = net(x)
        rec = out['x_hat'].clamp(0, 1)
        rec = transforms.ToPILImage()(rec.squeeze().cpu())
        rec.save('./images/infer.png', format="PNG")

        print('saved in ./images')