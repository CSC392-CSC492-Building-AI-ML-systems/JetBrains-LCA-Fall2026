```python
import argparse
from fate.ml.nn.hetero_nn import HeteroNNGuestModel, HeteroNNHostModel, HeteroNNModel, set_hetero_nn_model_meta
from fate.ml.pipeline import Pipeline
from fate.ml.components import (
    Reader,
    DataTransform,
    Intersection,
    Evaluation
)

def main(config_path):
    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    guest_data = config['guest_data']
    host_data = config['host_data']

    # Define pipeline
    pipeline = Pipeline()

    # Reader components
    reader_guest = Reader(name="reader_guest", role="guest", data=guest_data)
    reader_host = Reader(name="reader_host", role="host", data=host_data)

    # DataTransform components
    data_transform_guest = DataTransform(name="data_transform_guest", role="guest", with_label=True, output_format="dense")
    data_transform_host = DataTransform(name="data_transform_host", role="host", with_label=False, output_format="dense")

    # Intersection component
    intersect = Intersection(name="intersect")

    # HeteroNN configuration
    hetero_nn_guest = HeteroNNGuestModel(
        epochs=10,
        learning_rate=0.01,
        batch_size=64,
        interactive_layer_lr=0.01,
        interactive_layer_epoch=10,
        guest_bottom_model={"units": [64, 32], "activation": "relu"},
        guest_top_model={"units": [16, 1], "activation": ["relu", "sigmoid"]},
        interactive_layer={"units": 8, "activation": "relu"},
        callback_param={"early_stop": True, "patience": 3}
    )
    hetero_nn_host = HeteroNNHostModel(
        host_bottom_model={"units": [64, 32], "activation": "relu"}
    )

    # Evaluation component
    evaluation = Evaluation(name="evaluation", eval_type="binary")

    # Add components to pipeline
    pipeline.add_component(reader_guest)
    pipeline.add_component(reader_host)
    pipeline.add_component(data_transform_guest, data=reader_guest.output.data)
    pipeline.add_component(data_transform_host, data=reader_host.output.data)
    pipeline.add_component(intersect, data=[data_transform_guest.output.data, data_transform_host.output.data])
    pipeline.add_component(hetero_nn_guest, data=intersect.output.data)
    pipeline.add_component(hetero_nn_host, data=intersect.output.data)
    pipeline.add_component(evaluation, data=hetero_nn_guest.output.data)

    # Compile and fit pipeline
    pipeline.compile()
    pipeline.fit()

    # Print HeteroNN summary
    print(hetero_nn_guest.get_summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FATE HeteroNN Binary Classification Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    main(args.config)
```