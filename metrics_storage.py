import re
import torch
from torchmetrics import RetrievalMAP


class MetricsStorage:
    def __init__(self, metrics, database_embeddings, database_labels, query_embeddings, query_labels):
        # Compile pattern for P@k
        p_at_k_pattern = re.compile("P@[1-9][0-9]*$")

        # Calculate metrics
        self.metrics = {}
        ranking_metrics = []
        for metric in metrics:
            if metric == 'Sparsity':
                self.metrics['Sparsity'] = torch.cat(
                    (self.calculate_sparsity(database_embeddings), self.calculate_sparsity(query_embeddings))
                )
            elif metric == 'Independence':
                self.metrics['Independence'] = torch.cat(
                    (self.calculate_independence(database_embeddings), self.calculate_independence(query_embeddings))
                )
            elif p_at_k_pattern.match(metric) or metric == 'MAP':
                ranking_metrics.append(metric)
            else:
                raise Exception(f"Unknown metric: {metric}")

        if ranking_metrics:
            self.calculate_ranking_metrics(
                database_embeddings, database_labels, query_embeddings, query_labels, ranking_metrics
            )

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

    def calculate_ranking_metrics(self, database_embeddings, database_labels, query_embeddings, query_labels, ranking_metrics):
        # For each query embedding calculate distances to embeddings in database
        dists = torch.cdist(query_embeddings, database_embeddings)

        # Sort each row in ascending order
        sorted_dists, sorted_dist_inds = torch.sort(dists, dim=1)

        # Using indices above, get labels of database embeddings for each row in the order of increasing distance
        closest_database_labels = database_labels[sorted_dist_inds]

        # Now for each query (row) we can find if recommendations have the same label
        correct_labels = torch.eq(closest_database_labels, query_labels.unsqueeze(dim=1))

        for metric in ranking_metrics:
            if metric == 'MAP':
                # Find relevance of each recommendation. We will define it as: 1 / embedding_dist * min_embedding_dist
                # This way the closest embedding will have relevance 1, and relevance will decrease with larger distance
                # Since dists are already sorted, the smallest dist is the first
                relevance = sorted_dists[:, 0].unsqueeze(dim=1) / sorted_dists

                # Index of query for which recommendation is given
                # It is needed for MAP, because it asks to flatten all recommendations. So we need to tell, which
                # recommendation corresponds to which query (specifics of realization of RetrievalMAP)
                indexes = torch.tensor(
                    [query_ind for query_ind in range(len(query_labels)) for _ in range(len(database_labels))]
                )

                # Calculate MAP
                self.metrics[metric] = RetrievalMAP()(
                    relevance.flatten(), correct_labels.flatten(), indexes
                )
            else:
                # Get k from P@k
                k = int(metric[2:])

                # Find precision by taking labels of the first k columns (they correspond to closets k embeddings to
                # the queries) and summing bools (true if the same label)
                self.metrics[metric] = correct_labels[:, :k].sum(dim=1) / k

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
