import torch
from torch import nn
import torch.nn.functional as F
from utils.nn import mlp, clamp_probs, gumbel_softmax

from settings import args
from model.dirichlet import DirichletProcess


class DependencyAwareGeneration(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.w_cls = args.w_cls
        self.w_rec = args.w_rec
        nr_class = args.nr_class

        self.prediction = DirichletProcess(nr_class, L)
        self.inf = mlp([L+nr_class, 256, 512, 256], batchNorm=True, dropout_ratio=args.drop_ratio, nonlinearity='relu')
        self.inf_fc = mlp([256, nr_class], True)
        self.gen = mlp([L+nr_class, 256, 512], batchNorm=True, dropout_ratio=args.drop_ratio, nonlinearity='relu')
        self.gen_fc = mlp([512, nr_class], True)

    def inference_encode(self, x, y_priv):
        result = self.inf(torch.cat((x, y_priv), dim=1))
        result = torch.softmax(self.inf_fc(result), dim=1)
        return result

    def inference_decode(self, x, gt_labels):
        result = self.gen(torch.cat([x, gt_labels], dim=1))
        result = torch.softmax(self.gen_fc(result), dim=1)
        return result

    def forward(self, x, y_priv):
        y_hat = self.prediction.infer(x)
        gt_convidence = self.inference_encode(x, y_priv)
        gt_targets = gumbel_softmax(gt_convidence)
        rec_convidence = self.inference_decode(x, gt_targets)
        return y_hat, gt_convidence, rec_convidence

    def cal_loss(self, y_hat, gt_convidence, rec_convidence, y_priv):
        y_pred = torch.softmax(y_hat, dim=1)
        cls_loss = F.cross_entropy(y_pred, rec_convidence.detach())
        gt_convidence = clamp_probs(gt_convidence)
        ent_loss = -torch.mean(
            gt_convidence * torch.log(gt_convidence) + (1 - gt_convidence) * torch.log1p(-gt_convidence))
        rec_loss = F.cross_entropy(rec_convidence, y_priv)
        loss = self.w_rec * rec_loss + self.w_cls * (cls_loss - ent_loss)
        results = {
            'cls_loss': cls_loss.item(),
            'rec_loss': rec_loss.item(),
            'ent_loss': ent_loss.item()
        }
        return loss, results

    def cal_objective(self, x, y_priv):
        y_hat, gt_conv, rec_conv = self(x, y_priv)
        loss, res = self.cal_loss(y_hat, gt_conv, rec_conv, y_priv)
        return loss, res

    def predict(self, x, y):
        y_hat, _, _ = self(x, y)
        y_pred = torch.softmax(y_hat, dim=1)
        return y_pred

