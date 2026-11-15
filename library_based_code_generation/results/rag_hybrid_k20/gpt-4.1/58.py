```python
import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torchrec.models.dlrm import DLRM
from torchrec.optim.fused import FusedOptimizer
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.train_pipeline import TrainPipelineSparseDist
from torchrec.datasets.random import RandomRecDataset

def _get_random_dataloader(num_embeddings: int, batch_size: int, pin_memory: bool) -> DataLoader:
    dataset = RandomRecDataset(
        keys=["sparse"],
        batch_size=batch_size,
        num_dense=13,
        num_sparse=26,
        num_embeddings=num_embeddings,
        hash_sizes=[num_embeddings] * 26,
        ids_per_feature=1,
        num_batches=100,
    )
    return DataLoader(
        dataset,
        batch_size=None,
        pin_memory=pin_memory,
    )

def train(
    num_embeddings: int,
    embedding_dim: int,
    dense_arch_layer_sizes: list,
    over_arch_layer_sizes: list,
    learning_rate: float,
):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    model = DLRM(
        embedding_bag_collection=None,  # Should be constructed as per use-case
        dense_in_features=13,
        dense_arch_layer_sizes=dense_arch_layer_sizes,
        over_arch_layer_sizes=over_arch_layer_sizes,
        embedding_dim=embedding_dim,
    ).to(device)

    optimizer = FusedOptimizer(
        model.parameters(),
        lr=learning_rate,
    )

    model = DistributedModelParallel(model, device_ids=[rank])

    dataloader = _get_random_dataloader(num_embeddings, batch_size=1024, pin_memory=True)

    train_pipeline = TrainPipelineSparseDist(
        module=model,
        optimizer=optimizer,
        dataloader=dataloader,
        device=device,
        overlap_communication=True,
        overlap_compute=True,
        overlap_transfer=True,
    )

    for epoch in range(1):
        for _ in range(100):
            train_pipeline.progress()

if __name__ == "__main__":
    train(
        num_embeddings=10000,
        embedding_dim=64,
        dense_arch_layer_sizes=[512, 256, 64],
        over_arch_layer_sizes=[512, 256, 1],
        learning_rate=0.01,
    )
```