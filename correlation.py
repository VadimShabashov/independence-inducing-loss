import torch


def non_diagonal_correlation(x):
    """
    Calculation of the mean for non-diagonal elements of correlation matrix.
    """

    # Calculate correlation matrix
    correlation_matrix = torch.corrcoef(x.T)

    # Create mask for diagonal elements
    diagonal_elements_mask = torch.eye(correlation_matrix.shape[0], dtype=torch.bool)

    # Get non-diagonal elements
    non_diagonal_elements = correlation_matrix[~diagonal_elements_mask]

    return non_diagonal_elements.abs().mean()
