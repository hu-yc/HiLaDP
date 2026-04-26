import os
import torch

from utils.dataloader import load_disk_data
from utils.fea_extractor import extract_fea
from model.prior import DependenceAwarePrior
from settings import args, seed_everything, logging, device


def load_extracted_dataset():
    logging.debug("Loading the raw dataset")
    raw_train_loader, raw_test_loader = load_disk_data(args.ds)
    logging.debug("Extracting features from the dataset")
    train_dataset = extract_fea(raw_train_loader)
    test_dataset = extract_fea(raw_test_loader)
    train_data, train_labels = train_dataset.tensors
    test_data, test_labels = test_dataset.tensors
    data = torch.cat([train_data, test_data], dim=0).to(device)
    labels = torch.cat([train_labels, test_labels], dim=0).to(device)
    train_test_split = len(train_data) / (len(train_data) + len(test_data))
    return data, labels, train_test_split

def group_features_labels(x, y, unique_y):
    features = []
    for i_y in unique_y:
        features.append(x[torch.argmax(y, dim=1) == i_y])
    return features

def calculate_prior(features, prior_matrix):
    one_hot_labels = torch.eye(args.nr_class, device=device)
    cluster_train_data, cluster_train_lab, cluster_test_data, cluster_test_lab = [], [], [], []
    prior_train, prior_test = [], []
    for lab_idx in unique_labels:
        num_ins = features[lab_idx].shape[0]
        i_lab = one_hot_labels[lab_idx, :].unsqueeze(0).expand(num_ins, -1)
        split_idx = int(num_ins * split_ratio)
        cluster_train_data.append(features[lab_idx][:split_idx, :])
        cluster_test_data.append(features[lab_idx][split_idx:, :])
        cluster_train_lab.append(i_lab[:split_idx, :])
        cluster_test_lab.append(i_lab[split_idx:, :])
        prior_train.append(prior_matrix[lab_idx][:split_idx, :])
        prior_test.append(prior_matrix[lab_idx][split_idx:, :])
    cluster_train_data, cluster_train_lab, \
        cluster_test_data, cluster_test_lab, prior_train, prior_test = torch.cat(cluster_train_data, dim=0).cpu(), \
        torch.cat(cluster_train_lab, dim=0).cpu(), \
        torch.cat(cluster_test_data, dim=0).cpu(), \
        torch.cat(cluster_test_lab, dim=0).cpu(), \
        torch.softmax(torch.cat(prior_train, dim=0), dim=1).cpu(), \
        torch.softmax(torch.cat(prior_test, dim=0), dim=1).cpu()
    cluster_train_dataset = torch.utils.data.TensorDataset(cluster_train_data, cluster_train_lab)
    cluster_test_dataset = torch.utils.data.TensorDataset(cluster_test_data, cluster_test_lab)
    return cluster_train_dataset, cluster_test_dataset, prior_train, prior_test



if __name__ == '__main__':
    seed_everything(args.seed)
    logging.info("Loading {} datasets".format(args.ds))
    # Extracting features in dataset
    data_dir = os.path.join('data', args.ds)
    os.makedirs(data_dir, exist_ok=True)

    logging.info("Loading extracted features for the dataset")
    data, labels, split_ratio = load_extracted_dataset()

    unique_labels = torch.unique(torch.argmax(labels, dim=1), dim=0)
    grouped_features = group_features_labels(data, labels, unique_labels)

    model = DependenceAwarePrior(n_classes=args.nr_class, trunc=args.trunc, eta=args.eta, dim=args.nr_fea)
    model.train(grouped_features, epochs=args.epochs, lr=args.lr, bs=args.bs)
    cluster_assignments, cluster_probs = model.cluster_data(grouped_features)
    centroids = model.calculate_cluster_centroids(grouped_features, cluster_assignments)
    similarity_matrix = model.calculate_instance_class_similarity(centroids, cluster_assignments, grouped_features)

    train_dataset, test_dataset, prior_train, _ = calculate_prior(grouped_features, similarity_matrix)


    torch.save(train_dataset, os.path.join(data_dir, "train_datasets.pt"))
    torch.save(test_dataset, os.path.join(data_dir, "test_datasets.pt"))
    torch.save(prior_train, os.path.join(data_dir, "label_prior_train.pt"))
