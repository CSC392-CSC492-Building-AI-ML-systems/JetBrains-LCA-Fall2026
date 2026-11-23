```python
import argparse
import torch
import torch.nn as nn
from fate.ml.nn.hetero_nn import HeteroNNGuestModel, HeteroNNHostModel, set_hetero_nn_model_meta
from fate.ml.pipeline import Pipeline
from fate.ml.components import Reader, DataTransform, Intersection, HeteroNN, Evaluation

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

    def forward(self, x):
        return self.fc(x)

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
    guest_bottom = GuestBottomModel(guest_input_dim, hidden_dim)
    guest_top = GuestTopModel(interactive_dim, top_output_dim)
    host_bottom = HostBottomModel(host_input_dim, hidden_dim)
    interactive_layer = InteractiveLayer(hidden_dim * 2, interactive_dim)

    guest_model = HeteroNNGuestModel(
        bottom_model=guest_bottom,
        top_model=guest_top,
        interactive_layer=interactive_layer
    )
    host_model = HeteroNNHostModel(
        bottom_model=host_bottom
    )

    set_hetero_nn_model_meta(
        guest_model=guest_model,
        host_model=host_model,
        optimizer=torch.optim.Adam,
        loss_func=nn.BCEWithLogitsLoss(),
        learning_rate=learning_rate
    )

    # Build pipeline
    pipeline = Pipeline()

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=config['guest_party_id']).component_param(
        table=guest_data_table
    )
    reader_0.get_party_instance(role='host', party_id=config['host_party_id']).component_param(
        table=host_data_table
    )

    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role='guest', party_id=config['guest_party_id']).component_param(
        with_label=True, output_format="dense"
    )
    data_transform_0.get_party_instance(role='host', party_id=config['host_party_id']).component_param(
        with_label=False
    )

    intersection_0 = Intersection(name="intersection_0")

    hetero_nn_0 = HeteroNN(name="hetero_nn_0")
    hetero_nn_0.get_party_instance(role='guest', party_id=config['guest_party_id']).component_param(
        epochs=epochs,
        batch_size=batch_size,
        task_type=task_type
    )
    hetero_nn_0.get_party_instance(role='host', party_id=config['host_party_id']).component_param()

    hetero_nn_0.set_model(guest_model=guest_model, host_model=host_model)

    evaluation_0 = Evaluation(name="evaluation_0")
    evaluation_0.get_party_instance(role='guest', party_id=config['guest_party_id']).component_param(
        eval_type="binary"
    )

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=reader_0.output.data)
    pipeline.add_component(intersection_0, data=data_transform_0.output.data)
    pipeline.add_component(hetero_nn_0, data=intersection_0.output.data)
    pipeline.add_component(evaluation_0, data=hetero_nn_0.output.data)

    pipeline.compile()
    pipeline.fit()

    print(hetero_nn_0.get_summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FATE HeteroNN Binary Classification Pipeline")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    args = parser.parse_args()
    main(args.config)
```