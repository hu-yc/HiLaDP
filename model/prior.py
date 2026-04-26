import torch
import torch.optim as optim
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, TensorDataset

from model.dirichlet import DirichletProcess
from settings import device, logging


class DependenceAwarePrior:
    def __init__(self, n_classes=10, trunc=20, eta=1.0, dim=512):
        self.n_classes = n_classes
        self.trunc = trunc
        self.dim = dim
        self.eta = eta
        self.device = device

        self.models = [DirichletProcess(trunc=self.trunc, dim=self.dim, eta=self.eta) for _ in range(n_classes)]
        for model in self.models:
            model.to(self.device)

    def train(self, class_features, epochs, lr, bs):
        optimizers = [optim.Adam(model.parameters(), lr=lr) for model in self.models]

        for class_idx in range(self.n_classes):
            if len(class_features[class_idx]) == 0:
                logging.warning(f"skip {class_idx} (no features)")
                continue

            logging.info(f"training class {class_idx}-th model")
            model = self.models[class_idx]
            optimizer = optimizers[class_idx]
            features = class_features[class_idx].to(self.device)

            dataset = TensorDataset(features)
            loader = DataLoader(dataset, batch_size=bs, shuffle=True)

            for epoch in range(epochs):
                total_loss = 0.0
                for (batch_features,) in loader:
                    optimizer.zero_grad()
                    loss = -model(batch_features)  
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

    def cluster_data(self, class_features):
        cluster_assignments = []
        cluster_probs = []

        for class_idx in range(self.n_classes):
            if len(class_features[class_idx]) == 0:
                cluster_assignments.append([])
                cluster_probs.append([])
                continue

            model = self.models[class_idx]
            features = class_features[class_idx].to(self.device)

            with torch.no_grad():
                logits = model.infer(features)
                probs = torch.softmax(logits, dim=1)
                assignments = torch.argmax(probs, dim=1)

            cluster_assignments.append(assignments)
            cluster_probs.append(probs)

        return cluster_assignments, cluster_probs

    def calculate_cluster_centroids(self, class_features, cluster_assignments):
        centroids = []

        for class_idx in range(self.n_classes):
            if len(class_features[class_idx]) == 0:
                centroids.append([])
                continue

            features = class_features[class_idx]
            assignments = cluster_assignments[class_idx]

            class_centroids = []
            for cluster_idx in range(self.trunc):
                cluster_samples = features[assignments == cluster_idx]
                if len(cluster_samples) > 0:
                    centroid = torch.mean(cluster_samples, axis=0)
                    class_centroids.append(centroid)
                else:
                    class_centroids.append(None)

            centroids.append(class_centroids)

        return centroids

    def calculate_class_similarity(self, centroids):
        similarity_matrix = torch.zeros((self.n_classes, self.n_classes))

        for i in range(self.n_classes):
            for j in range(i, self.n_classes):
                if len(centroids[i]) == 0 or len(centroids[j]) == 0:
                    similarity_matrix[i, j] = similarity_matrix[j, i] = 0
                    continue

                max_similarity = torch.zeros(1, device=self.device)
                valid_i_centroids = [c for c in centroids[i] if c is not None]
                valid_j_centroids = [c for c in centroids[j] if c is not None]

                if not valid_i_centroids or not valid_j_centroids:
                    similarity_matrix[i, j] = similarity_matrix[j, i] = 0
                    continue

                for ci in valid_i_centroids:
                    for cj in valid_j_centroids:
                        similarity = torch.dot(ci, cj) / (torch.norm(ci) * torch.norm(cj))
                        max_similarity = torch.max(max_similarity, similarity)

                similarity_matrix[i, j] = similarity_matrix[j, i] = max_similarity
        return similarity_matrix

    
    def calculate_instance_class_similarity(self, centroids, cluster_assignments, class_features):
        n_classes = self.n_classes
        instance_similarities = []
        for class_idx in range(n_classes):
            if len(class_features[class_idx]) == 0:
                instance_similarities.append(torch.tensor([]))
                continue
            n_instances = len(class_features[class_idx])
            similarity_matrix = torch.zeros((n_instances, n_classes), device=self.device)
            assignments = cluster_assignments[class_idx]
            valid_clusters = torch.tensor([i for i, c in enumerate(centroids[class_idx]) 
                                        if c is not None], device=self.device)
            valid_mask = torch.tensor([a in valid_clusters for a in assignments], device=self.device)
            valid_instances = torch.where(valid_mask)[0]
            if len(valid_instances) == 0:
                logging.warning(f"There is not valida instances in the class {class_idx}")
                instance_similarities.append(similarity_matrix)
                continue
            valid_assignments = assignments[valid_instances]
            instance_centroids = torch.stack([centroids[class_idx][a.item()] for a in valid_assignments])
            for target_class_idx in range(n_classes):
                if len(centroids[target_class_idx]) == 0:
                    continue
                valid_target_centroids = [c for c in centroids[target_class_idx] if c is not None]
                if len(valid_target_centroids) == 0:
                    continue
                target_centroids_tensor = torch.stack(valid_target_centroids)
                normalized_instance = normalize(instance_centroids, p=2, dim=1)
                normalized_target = normalize(target_centroids_tensor, p=2, dim=1)
                sim = torch.mm(normalized_instance, normalized_target.t())
                max_sim, _ = torch.max(sim, dim=1)  # [n_valid_instances]
                similarity_matrix[valid_instances, target_class_idx] = max_sim
            instance_similarities.append(similarity_matrix)
        return instance_similarities

    def generate_enhanced_features(self, class_features, cluster_assignments, centroids):
        enhanced_features = []

        for class_idx in range(self.n_classes):
            if len(class_features[class_idx]) == 0:
                enhanced_features.append([])
                continue

            orig_features = class_features[class_idx]
            assignments = cluster_assignments[class_idx]
            class_centroids = centroids[class_idx]

            assigned_centroids = []
            for i, cluster_idx in enumerate(assignments):
                if cluster_idx < len(class_centroids) and class_centroids[cluster_idx] is not None:
                    assigned_centroids.append(class_centroids[cluster_idx])
                else:
                    assigned_centroids.append(torch.zeros_like(orig_features[i]))
            assigned_centroids = torch.stack(assigned_centroids)
            new_features = torch.cat([orig_features, assigned_centroids], dim=1)
            enhanced_features.append(new_features)

        return enhanced_features
