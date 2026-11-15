```python
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchrec.datasets.random import RandomRecDataset
from torchrec.models.dlrm import DLRM
from torchrec.optim.fused import FusedOptimizer
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.train_pipeline import TrainPipelineSparseDist
from torchrec.distributed.train_pipeline import train_one_epoch_fused_optimizer

def _get_random_dataloader(num_embeddings: int, batch_size: int, pin_memory: bool) -> DataLoader:
    dataset = RandomRecDataset(
        keys=["f" + str(i) for i in range(num_embeddings)],
        batch_size=batch_size,
        hash_sizes=[10000] * num_embeddings,
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
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    model = DLRM(
        embedding_bag_collection_kwargs={
            "tables": [
                {
                    "num_embeddings": 10000,
                    "embedding_dim": embedding_dim,
                    "name": f"f{i}",
                    "feature_names": [f"f{i}"],
                }
                for i in range(num_embeddings)
            ]
        },
        dense_in_features=13,
        dense_arch_layer_sizes=dense_arch_layer_sizes,
        over_arch_layer_sizes=over_arch_layer_sizes,
    ).to(device)

    optimizer = FusedOptimizer(
        params=model.parameters(),
        lr=learning_rate,
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

    for epoch in range(1):
        train_one_epoch_fused_optimizer(train_pipeline, epoch=epoch)

if __name__ == "__main__":
    train(
        num_embeddings=4,
        embedding_dim=16,
        dense_arch_layer_sizes=[64, 32],
        over_arch_layer_sizes=[32, 16, 1],
        learning_rate=0.01,
    )
```