import re
import torch
from collections import defaultdict


class MetricsStorage:
    def __init__(self, metrics, database_embs, database_labels, query_embs, query_labels):
        # Compile pattern for P@k
        p_at_k_pattern = re.compile("P@[1-9][0-9]*$")

        # Calculate metrics
        self.metrics = defaultdict(list)
        p_at_k_metrics = []
        for metric in metrics:
            if metric == 'Sparsity':
                self.metrics['Sparsity'] = torch.cat(
                    (self.calculate_sparsity(database_embs), self.calculate_sparsity(query_embs)))
            elif metric == 'Independence':
                self.metrics['Independence'] = torch.cat(
                    (self.calculate_independence(database_embs), self.calculate_independence(query_embs)))
            elif p_at_k_pattern.match(metric):
                p_at_k_metrics.append(metric)
            else:
                raise Exception(f"Unknown metric: {metric}")

        if p_at_k_metrics:
            self.calculate_ranking_metric(database_embs, database_labels, query_embs, query_labels, p_at_k_metrics)

    @staticmethod
    def calculate_sparsity(embeddings):
        return (embeddings < 1e-8).to(torch.float32).mean(dim=1)

    @staticmethod
    def calculate_independence(embeddings):
        # Calculate correlation matrix
        correlation_matrix = torch.corrcoef(embeddings.T)

        # Replace nan with 0 (nan appears due to constant value -> zero variance)
        correlation_matrix[torch.isnan(correlation_matrix)] = 0.0

        # Find non-diagonal elements of correlation matrix
        non_diagonal_correlation_matrix = torch.abs(correlation_matrix) - torch.eye(correlation_matrix.shape[0])

        # For each feature find mean over its correlation with others
        return non_diagonal_correlation_matrix.mean(dim=1)

    def calculate_ranking_metric(self, database_embs, database_labels, query_embs, query_labels, p_at_k_metrics):
        # For each query embedding calculate distances to embeddings in database
        dists = torch.cdist(query_embs, database_embs)

        # Sort each row in ascending order. Get indices, where each distance was in initial row
        sorted_dist_inds = torch.argsort(dists, dim=1)

        # Get labels for each row by indices obtained above
        # Idea: row ind corresponds to query and after this step for each query we would have list of labels
        # in the order of increasing distance
        closest_database_labels = database_labels[sorted_dist_inds]

        # Now for each query (row) we can find if recommendations have the same label
        correct_labels = closest_database_labels == query_labels.unsqueeze(dim=1)

        for p_at_k in p_at_k_metrics:
            # Get k from p@k
            k = int(p_at_k[2:])

            # Find precision by taking first k columns (they correspond to closets embeddings to the queries)
            # and summing bools (true if the same label)
            self.metrics[p_at_k] = correct_labels[:, :k].sum(dim=1) / k

    def get_metric(self, metric):
        if metric in self.metrics:
            return self.metrics[metric]
        else:
            raise Exception(f"Unknown metric '{metric}'")

    def get_mean_of_metrics(self, track_metrics):
        # List of tracking metrics means
        means_track_metrics = []
        for metric in track_metrics:
            means_track_metrics.append(self.metrics[metric].mean().item())

        return means_track_metrics
