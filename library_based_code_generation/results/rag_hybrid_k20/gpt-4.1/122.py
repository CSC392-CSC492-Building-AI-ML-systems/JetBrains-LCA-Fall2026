```python
import argparse
import torch
import torch.nn as nn
from fate.arch import Context
from fate.ml.nn.hetero_nn import HeteroNN, HeteroNNGuestModel, HeteroNNHostModel
from fate.ml.pipeline import Pipeline
from fate.ml.components import Reader, DataTransform, Intersection, Evaluation

class GuestBottomModel(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class GuestTopModel(nn.Module):
    def __init__(self, input_dim=64, output_dim=1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

class HostBottomModel(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

def main(config_path):
    # Load config
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    guest_train_data = config["guest_train_data"]
    host_train_data = config["host_train_data"]
    guest_data_table = guest_train_data["name"]
    guest_data_namespace = guest_train_data["namespace"]
    host_data_table = host_train_data["name"]
    host_data_namespace = host_train_data["namespace"]

    guest_input_dim = config["guest_input_dim"]
    host_input_dim = config["host_input_dim"]

    epochs = config.get("epochs", 10)
    learning_rate = config.get("learning_rate", 0.01)
    batch_size = config.get("batch_size", 64)
    task_type = config.get("task_type", "binary")

    # Pipeline
    pipeline = Pipeline()

    # Reader
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role="guest", party_id=0).component_param(
        table={"name": guest_data_table, "namespace": guest_data_namespace}
    )
    reader_0.get_party_instance(role="host", party_id=0).component_param(
        table={"name": host_data_table, "namespace": host_data_namespace}
    )

    # DataTransform
    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role="guest", party_id=0).component_param(
        with_label=True, output_format="dense"
    )
    data_transform_0.get_party_instance(role="host", party_id=0).component_param(
        with_label=False
    )

    # Intersection
    intersection_0 = Intersection(name="intersection_0")

    # HeteroNN
    hetero_nn_0 = HeteroNN(name="hetero_nn_0")
    # Guest models
    guest_bottom_model = GuestBottomModel(input_dim=guest_input_dim)
    guest_top_model = GuestTopModel(input_dim=64)
    # Host model
    host_bottom_model = HostBottomModel(input_dim=host_input_dim)

    hetero_nn_0.get_party_instance(role="guest", party_id=0).component_param(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        task_type=task_type,
        guest_bottom_model=HeteroNNGuestModel(guest_bottom_model),
        guest_top_model=HeteroNNGuestModel(guest_top_model),
        interactive_layer="concat",
        optimizer=torch.optim.Adam,
        loss_fn=nn.BCELoss()
    )
    hetero_nn_0.get_party_instance(role="host", party_id=0).component_param(
        host_bottom_model=HeteroNNHostModel(host_bottom_model)
    )

    # Evaluation
    evaluation_0 = Evaluation(name="evaluation_0")
    evaluation_0.get_party_instance(role="guest", party_id=0).component_param(
        eval_type="binary"
    )

    # Add components to pipeline
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=reader_0.output.data)
    pipeline.add_component(intersection_0, data=data_transform_0.output.data)
    pipeline.add_component(hetero_nn_0, data=intersection_0.output.data)
    pipeline.add_component(evaluation_0, data=hetero_nn_0.output.data)

    # Compile and fit
    pipeline.compile()
    pipeline.fit()

    # Print summary
    summary = pipeline.get_component("hetero_nn_0").get_summary()
    print("HeteroNN Summary:", summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)
```