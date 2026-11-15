```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torchrec.datasets.random import RandomRecDataset
from torchrec.models.dlrm import DLRM
from torchrec.distributed import DistributedModelParallel
from torchrec.optim.fused import FusedOptimizer
from torchrec.distributed.train_pipeline import TrainPipelineSparseDist
from torchrec.distributed.embedding_types import EmbeddingComputeKernel

def _get_random_dataloader(num_embeddings: int, batch_size: int, pin_memory: bool) -> DataLoader:
    dataset = RandomRecDataset(
        keys=["f" + str(i) for i in range(26)],
        batch_size=batch_size,
        hash_sizes=[num_embeddings] * 26,
        ids_per_feature=1,
        num_dense=13,
        num_batches=100,
    )
    return DataLoader(dataset, batch_size=None, pin_memory=pin_memory)

def train(
    num_embeddings: int,
    embedding_dim: int,
    dense_arch_layer_sizes: list,
    over_arch_layer_sizes: list,
    learning_rate: float,
):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    embedding_tables = [
        {
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "name": f"t_{i}",
            "feature_names": [f"f{i}"],
        }
        for i in range(26)
    ]

    model = DLRM(
        embedding_tables=embedding_tables,
        dense_in_features=13,
        dense_arch_layer_sizes=dense_arch_layer_sizes,
        over_arch_layer_sizes=over_arch_layer_sizes,
        dense_device=device,
        embedding_device=device,
        embedding_kernel=EmbeddingComputeKernel.FUSED,
    ).to(device)

    optimizer = FusedOptimizer(
        optim.SGD(model.parameters(), lr=learning_rate)
    )

    model = DistributedModelParallel(model, device=device)

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

    for batch in train_pipeline:
        loss = batch["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

if __name__ == "__main__":
    train(
        num_embeddings=10000,
        embedding_dim=64,
        dense_arch_layer_sizes=[512, 256, 64],
        over_arch_layer_sizes=[512, 256, 1],
        learning_rate=0.01,
    )
```