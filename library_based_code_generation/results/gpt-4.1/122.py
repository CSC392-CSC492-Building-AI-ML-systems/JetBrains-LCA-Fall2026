```python
import argparse
import torch
import torch.nn as nn
from fate.arch import Context
from fate.ml.nn.homo import TorchModel
from fate.ml.nn.hetero import HeteroNN
from fate.ml.pipeline import Pipeline
from fate.ml.components import Reader, DataTransform, Intersection, Evaluation

class GuestBottomModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GuestBottomModel, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class GuestTopModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GuestTopModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

class HostBottomModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(HostBottomModel, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class InteractiveLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InteractiveLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

def main(config_path):
    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    guest_train_data = config['guest_train_data']
    host_train_data = config['host_train_data']
    guest_data_table = guest_train_data['name'], guest_train_data['namespace']
    host_data_table = host_train_data['name'], host_train_data['namespace']

    guest_input_dim = config['guest_input_dim']
    host_input_dim = config['host_input_dim']
    hidden_dim = config['hidden_dim']
    interactive_dim = config['interactive_dim']
    top_output_dim = config['top_output_dim']

    epochs = config['epochs']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    task_type = config['task_type']

    # Define models
    guest_bottom_model = GuestBottomModel(guest_input_dim, hidden_dim)
    guest_top_model = GuestTopModel(interactive_dim, top_output_dim)
    host_bottom_model = HostBottomModel(host_input_dim, hidden_dim)
    interactive_layer = InteractiveLayer(hidden_dim * 2, interactive_dim)

    # Wrap models with TorchModel
    guest_bottom = TorchModel(guest_bottom_model)
    guest_top = TorchModel(guest_top_model)
    host_bottom = TorchModel(host_bottom_model)
    interactive = TorchModel(interactive_layer)

    # Build pipeline
    pipeline = Pipeline()

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=0).component_param(
        table=guest_data_table
    )
    reader_0.get_party_instance(role='host', party_id=0).component_param(
        table=host_data_table
    )

    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role='guest', party_id=0).component_param(
        with_label=True, output_format="dense"
    )
    data_transform_0.get_party_instance(role='host', party_id=0).component_param(
        with_label=False
    )

    intersection_0 = Intersection(name="intersection_0")

    hetero_nn_0 = HeteroNN(name="hetero_nn_0",
                           epochs=epochs,
                           batch_size=batch_size,
                           learning_rate=learning_rate,
                           task_type=task_type,
                           interactive_layer=interactive,
                           guest_bottom_model=guest_bottom,
                           guest_top_model=guest_top,
                           host_bottom_model=host_bottom,
                           optimizer=torch.optim.Adam,
                           loss=nn.BCELoss())

    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=reader_0.output.data)
    pipeline.add_component(intersection_0, data=data_transform_0.output.data)
    pipeline.add_component(hetero_nn_0, data=intersection_0.output.data)
    pipeline.add_component(evaluation_0, data=hetero_nn_0.output.data)

    pipeline.compile()
    pipeline.fit()

    summary = pipeline.get_component("hetero_nn_0").get_summary()
    print(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FATE HeteroNN Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    main(args.config)
```