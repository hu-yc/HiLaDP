import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torchvision.models import resnet18, ResNet18_Weights

from settings import args, device


class PreTrainedExtractor(nn.Module):
    def __init__(self, params=args):
        super(PreTrainedExtractor, self).__init__()
        self.args = params
        self.ds_name = self.args.ds
        self.nr_class = self.args.nr_class
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, self.nr_class)
        )
        pretrained_pth = os.path.join('pretrain', 'resnet-18_{}.pth'.format(args.ds)) # output (bs, 512, 1, 1)
        model.load_state_dict(torch.load(pretrained_pth, weights_only=True))
        model.eval()
        self.pretrained_ext = nn.Sequential(*list(model.children())[:-1]) 

    def forward(self, x):
        x = self.pretrained_ext(x)
        x = x.view(x.size(0), -1)
        return x


def extract_fea(data_loader):
    xs_list = []
    y_list = []
    loader_iter = tqdm(data_loader)
    fea_extractor = PreTrainedExtractor().to(device)
    with torch.no_grad():
        for x, y in loader_iter:
            x = x.to(device)
            xs = fea_extractor(x)
            xs_list.append(xs.detach().cpu())
            y_list.append(y)
    new_xs = torch.cat(xs_list, dim=0)
    y = torch.cat(y_list, dim=0)
    dataset = TensorDataset(torch.tensor(new_xs), torch.tensor(y))
    return dataset
