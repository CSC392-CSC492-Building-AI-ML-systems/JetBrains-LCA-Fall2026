```python
import argparse
import torch
import torch.nn as nn
from fate.ml.nn.model_zoo.hetero_nn import HeteroNN, HeteroNNParam
from fate.ml.nn.model_zoo.hetero_nn import HeteroNNGuest, HeteroNNHost
from fate.ml.nn.model_zoo.hetero_nn import HeteroNNGuestModel, HeteroNNHostModel
from fate.ml.nn.model_zoo.hetero_nn import TorchNNModel
from fate.ml.pipeline import Pipeline
from fate.ml.pipeline.component import Reader, DataTransform, Intersection, Evaluation
from fate.ml.pipeline.interface import Data
from fate.ml.pipeline.utils import load_job_config

# Define guest bottom model
class GuestBottomModel(nn.Module):
    def __init__(self, input_dim):
        super(GuestBottomModel, self).__init__()
        self.fc = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

# Define guest top model
class GuestTopModel(nn.Module):
    def __init__(self):
        super(GuestTopModel, self).__init__()
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

# Define host bottom model
class HostBottomModel(nn.Module):
    def __init__(self, input_dim):
        super(HostBottomModel, self).__init__()
        self.fc = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

# Define interactive layer
class InteractiveLayer(nn.Module):
    def __init__(self):
        super(InteractiveLayer, self).__init__()
        self.fc = nn.Linear(128, 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

def main(config_path):
    # Load config
    config = load_job_config(config_path)
    guest_train_data = config['guest']['train_data']
    host_train_data = config['host']['train_data']
    guest_data_table = guest_train_data['name'], guest_train_data['namespace']
    host_data_table = host_train_data['name'], host_train_data['namespace']

    # Pipeline
    pipeline = Pipeline()

    # Reader
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=config['guest']['party_id']).component_param(
        table=guest_data_table
    )
    reader_0.get_party_instance(role='host', party_id=config['host']['party_id']).component_param(
        table=host_data_table
    )

    # DataTransform
    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role='guest', party_id=config['guest']['party_id']).component_param(
        with_label=True, output_format="dense"
    )
    data_transform_0.get_party_instance(role='host', party_id=config['host']['party_id']).component_param(
        with_label=False
    )

    # Intersection
    intersection_0 = Intersection(name="intersection_0")

    # HeteroNN
    # Assume input_dim for guest and host are known or can be inferred from config
    guest_input_dim = config['guest'].get('input_dim', 30)
    host_input_dim = config['host'].get('input_dim', 20)

    guest_bottom_model = TorchNNModel(GuestBottomModel(guest_input_dim))
    guest_top_model = TorchNNModel(GuestTopModel())
    host_bottom_model = TorchNNModel(HostBottomModel(host_input_dim))
    interactive_layer = TorchNNModel(InteractiveLayer())

    hetero_nn_param = HeteroNNParam(
        epochs=5,
        batch_size=64,
        learning_rate=0.01,
        task_type='classification',
        interactive_layer_lr=0.01,
        config_type='nn',
        optimizer='Adam',
        loss='BCELoss'
    )

    hetero_nn_0 = HeteroNN(name="hetero_nn_0", param=hetero_nn_param)
    hetero_nn_0.add_guest_bottom_model(guest_bottom_model)
    hetero_nn_0.add_guest_top_model(guest_top_model)
    hetero_nn_0.add_host_bottom_model(host_bottom_model)
    hetero_nn_0.add_interactive_layer(interactive_layer)

    # Evaluation
    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    # Add components to pipeline
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_nn_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_nn_0.output.data))

    # Compile and fit
    pipeline.compile()
    pipeline.fit()

    # Print summary
    print(hetero_nn_0.get_summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FATE HeteroNN Pipeline")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config)
```