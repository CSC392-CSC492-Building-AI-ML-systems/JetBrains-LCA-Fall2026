```python
import argparse
from fate_flow.pipeline import Pipeline
from fate_flow.pipeline.components.federated import Reader, DataTransform, Intersection, Evaluation
from fate_flow.pipeline.components.federated import HeteroNN
from fate_flow.pipeline.interface import Data
from fate_flow.pipeline.utils.tools import load_job_config

def main(config_path):
    # Load configuration
    config = load_job_config(config_path)
    guest = config['role']['guest'][0]
    host = config['role']['host'][0]
    backend = config['backend']
    work_mode = config['work_mode']

    # Define data sources
    guest_train_data = {"name": config['data']['guest_train']['name'], "namespace": config['data']['guest_train']['namespace']}
    host_train_data = {"name": config['data']['host_train']['name'], "namespace": config['data']['host_train']['namespace']}

    # Components
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0", with_label=True, output_format="dense")
    data_transform_0.get_party_instance(role='guest', party_id=guest).component_param(label_name="y", label_type="int")
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(with_label=False)

    intersect_0 = Intersection(name="intersect_0")

    hetero_nn_0 = HeteroNN(
        name="hetero_nn_0",
        epochs=10,
        batch_size=64,
        learning_rate=0.01,
        interactive_layer_lr=0.01,
        interactive_layer_epoch=10,
        optimizer="Adam",
        early_stop={"early_stop": "diff", "eps": 1e-4},
        config_type="keras",
        guest_bottom_model={"units": [64, 32], "activation": "relu"},
        host_bottom_model={"units": [64, 32], "activation": "relu"},
        interactive_layer={"units": 64, "activation": "relu"},
        guest_top_model={"units": [32, 1], "activation": ["relu", "sigmoid"]},
        callback_param={"callbacks": ["EarlyStopping"], "patience": 3}
    )

    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    # Build pipeline
    pipeline = Pipeline()
    pipeline.set_initiator(role='guest', party_id=guest)
    pipeline.set_roles(guest=guest, host=host)

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersect_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_nn_0, data=Data(train_data=intersect_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_nn_0.output.data))

    # Compile and fit
    pipeline.compile()
    pipeline.fit()

    # Print summary
    print(hetero_nn_0.get_summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FATE HeteroNN Pipeline for Binary Classification")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()
    main(args.config)
```