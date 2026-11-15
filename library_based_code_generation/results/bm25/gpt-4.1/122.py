```python
import argparse
import torch
import torch.nn as nn
from fate.arch import Context
from fate.ml.nn.hetero_nn import HeteroNN
from fate.ml.nn.hetero_nn import GuestBottomModel, GuestTopModel, HostBottomModel, InteractiveLayer
from fate.ml.pipeline import Pipeline
from fate.ml.components import Reader, DataTransform, Intersection, Evaluation

class GuestBottom(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GuestBottom, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class GuestTop(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GuestTop, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

class HostBottom(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HostBottom, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class InteractLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InteractLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

def main(config_path):
    # Load config
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    guest_data = config["guest"]["train_data"]
    host_data = config["host"]["train_data"]
    guest_id = config["guest"]["party_id"]
    host_id = config["host"]["party_id"]

    # Pipeline
    pipeline = Pipeline()

    # Reader
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role="guest", party_id=guest_id).component_param(table=guest_data)
    reader_0.get_party_instance(role="host", party_id=host_id).component_param(table=host_data)

    # DataTransform
    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role="guest", party_id=guest_id).component_param(with_label=True, output_format="dense")
    data_transform_0.get_party_instance(role="host", party_id=host_id).component_param(with_label=False)

    # Intersection
    intersect_0 = Intersection(name="intersect_0")

    # HeteroNN
    hetero_nn_0 = HeteroNN(name="hetero_nn_0")
    hetero_nn_0.set_params(
        epochs=5,
        learning_rate=0.01,
        batch_size=64,
        task_type="classification"
    )

    # Define models
    guest_bottom_model = GuestBottom(input_dim=30, output_dim=16)
    guest_top_model = GuestTop(input_dim=16, output_dim=1)
    host_bottom_model = HostBottom(input_dim=20, output_dim=16)
    interactive_layer = InteractLayer(input_dim=32, output_dim=16)

    # Add models to HeteroNN
    hetero_nn_0.add_guest_bottom_model(guest_bottom_model)
    hetero_nn_0.add_guest_top_model(guest_top_model)
    hetero_nn_0.add_host_bottom_model(host_bottom_model)
    hetero_nn_0.add_interactive_layer(interactive_layer)

    # Compile HeteroNN
    optimizer = torch.optim.Adam
    loss_fn = nn.BCELoss()
    hetero_nn_0.compile(optimizer=optimizer, loss=loss_fn)

    # Evaluation
    eval_0 = Evaluation(name="eval_0", eval_type="binary")

    # Add components to pipeline
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=reader_0.output.data)
    pipeline.add_component(intersect_0, data=data_transform_0.output.data)
    pipeline.add_component(hetero_nn_0, data=intersect_0.output.data)
    pipeline.add_component(eval_0, data=hetero_nn_0.output.data)

    # Compile and fit
    pipeline.compile()
    pipeline.fit()

    # Print summary
    print(hetero_nn_0.get_summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FATE HeteroNN Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)
```