import os
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from model.generation import DependencyAwareGeneration
from settings import args, seed_everything, logging, device


def train():
    model.train()
    total_loss = 0
    total_cls = 0
    total_ent = 0
    total_rec = 0
    for x, y_priv in train_loader:
        x = x.to(torch.float32).to(device)
        y_priv = y_priv.to(torch.float32).to(device)
        optimizer.zero_grad()
        loss, res = model.cal_objective(x, y_priv)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_cls += res['cls_loss']
        total_ent += res['ent_loss']
        total_rec += res['rec_loss']
    total_loss /= len(train_loader)
    total_cls /= len(train_loader)
    total_ent /= len(train_loader)
    total_rec /= len(train_loader)
    return total_loss, total_cls, total_ent, total_rec


def evaluate():
    all_true_lab = []
    all_pred_lab = []
    y_pred_array = np.empty((0, args.nr_class))
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)
            y_pred = model.predict(x, y)
            _, pred_lab = torch.max(y_pred.data, 1)
            _, true_lab = torch.max(y.data, 1)
            y_pred_array = np.vstack((y_pred_array, y_pred.detach().cpu().numpy()))
            all_true_lab.append(true_lab)
            all_pred_lab.append(pred_lab)
    all_true_lab = np.array(torch.cat(all_true_lab).cpu().numpy())
    all_pred_lab = np.array(torch.cat(all_pred_lab).cpu().numpy())
    acc_value = accuracy_score(all_true_lab, all_pred_lab)
    return acc_value


def generate_private_label(y, prior, eps):
    B, K = prior.shape
    device = prior.device
    dtype = prior.dtype
    argmax_y = torch.argmax(y, dim=1).to(device)
    tmp = torch.exp(torch.tensor(-eps, dtype=dtype, device=device))
    prior_sorted, idx_sort = torch.sort(prior, dim=1, descending=True)
    cumsum = torch.cumsum(prior_sorted, dim=1)
    denom = 1 + tmp * torch.arange(1, K + 1, device=device, dtype=dtype)
    wks = cumsum / denom
    optim_k = torch.argmax(wks, dim=1) + 1 
    adjusted_prior = torch.full_like(prior, tmp) 
    norm_factors = 1 + (optim_k - 1).float() * tmp  
    rows = torch.arange(B, device=device)
    adjusted_prior[rows, argmax_y] = 1.0
    topk_mask = torch.zeros_like(prior, dtype=torch.bool)
    for i in range(B):
        topk_mask[i, idx_sort[i, :optim_k[i]]] = True
    adjusted_prior = adjusted_prior * topk_mask
    adjusted_prior = adjusted_prior / norm_factors.view(-1, 1)
    adjusted_prior = adjusted_prior / adjusted_prior.sum(dim=1, keepdim=True)
    rr_index = torch.multinomial(adjusted_prior, num_samples=1).squeeze(1)
    rr_labels = torch.zeros_like(y)
    rr_labels[rows, rr_index] = 1
    return rr_labels



if __name__ == '__main__':
    seed_everything(args.seed)
    # load datasets
    logging.info("Loading {} datasets".format(args.ds))
    # Extracting features in dataset
    data_dir = os.path.join('data', args.ds)
    os.makedirs(data_dir, exist_ok=True)
    if os.path.exists(os.path.join(data_dir, "train_datasets.pt")) and os.path.exists(
            os.path.join(data_dir, "test_datasets.pt")):
        logging.debug("Loading extracted features with pre-trained model for the dataset")
        train_dataset = torch.load(os.path.join(data_dir, "train_datasets.pt"))
        test_dataset = torch.load(os.path.join(data_dir, "test_datasets.pt"))

        ## generating the privacy labels
        train_features, train_labels = train_dataset.tensors
        train_prior_pth = os.path.join(data_dir, "label_prior_train.pt")
        if os.path.exists(train_prior_pth):
            logging.info("Loading labels prior")
            train_labels_prior = torch.load(train_prior_pth).to(device)
        else:
            train_labels_prior = torch.ones([len(train_features), args.nr_class], device=device) / args.nr_class
        logging.info("Generating the differentially private label with epsilon = {}".format(args.eps))
        train_priv_labels = generate_private_label(train_labels, train_labels_prior, eps=args.eps)
        train_dataset = torch.utils.data.TensorDataset(train_features, train_priv_labels)

        logging.info("Creating dataloader with the pre-processed dataset")
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
        model = DependencyAwareGeneration(train_dataset.tensors[0].shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        start_time = time.time()
        for epoch in range(args.epochs):
            loss_value, cls_value, ent_value, rec_value = train()
            logging.info("Epoch {} >>> Loss: {:.4f}, cls: {:.4f}, ent: {:.4f}, rec: {:.4f}  lr: {:.4f}".format(epoch + 1, 
                                                                                                               loss_value, 
                                                                                                               cls_value, 
                                                                                                               ent_value, 
                                                                                                               rec_value,
                                                                                                               optimizer.param_groups[-1]['lr']))
        acc = evaluate()
        logging.info("Test Accuracy: {:.4f}".format(acc))
        end_time = time.time()
    else:
        logging.info("Please run the dependence-aware prior module first")

