import torch


def corr(x_batch: torch.Tensor) -> torch.Tensor:
    if len(x_batch.shape) == 2:
        x_batch = x_batch.unsqueeze(0)
    # x_batch.shape = [batch_size, n_features, decoder_steps]

    n_features = x_batch.shape[1]
    indices = torch.triu_indices(n_features, n_features, 1)

    correlations = []
    for x in x_batch:
        # x.shape = [n_features, decoder_steps]
        correlation = torch.corrcoef(x)
        # correlation.shape = [n_features, n_features]
        correlation = correlation[indices[0], indices[1]]
        # correlation.shape = [bin(n_features, 2)]

        correlations.append(torch.nan_to_num(correlation))

    correlations = torch.stack(correlations)
    # correlations.shape = [batch_size, bin(n_features, 2)]

    return correlations
