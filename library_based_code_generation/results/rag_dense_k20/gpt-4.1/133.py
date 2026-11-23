```python
import argparse
from fate.arch import Context
from fate.ml.nn.hetero_nn import HeteroNN, HeteroNNParam
from fate.ml.nn.hetero_nn import GuestBottomModel, HostBottomModel, InteractiveLayer, GuestTopModel
from fate.ml.pipeline import Pipeline
from fate.ml.components import Reader, DataTransform, Intersection, Evaluation

def main(config_path):
    # Load configuration
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    guest_data = config["guest_data"]
    host_data = config["host_data"]
    epochs = config.get("epochs", 10)
    learning_rate = config.get("learning_rate", 0.01)
    batch_size = config.get("batch_size", 64)
    callback_params = config.get("callback_params", {"early_stop": True, "patience": 3})

    # Define pipeline
    pipeline = Pipeline()

    # Reader components
    reader_guest = Reader(name="reader_guest")
    reader_guest.get_party_instance(role="guest", party_id=config["guest_id"]).component_param(
        table=guest_data["table"], namespace=guest_data["namespace"]
    )

    reader_host = Reader(name="reader_host")
    reader_host.get_party_instance(role="host", party_id=config["host_id"]).component_param(
        table=host_data["table"], namespace=host_data["namespace"]
    )

    # DataTransform components
    data_transform_guest = DataTransform(name="data_transform_guest")
    data_transform_guest.get_party_instance(role="guest", party_id=config["guest_id"]).component_param(
        with_label=True, output_format="dense"
    )

    data_transform_host = DataTransform(name="data_transform_host")
    data_transform_host.get_party_instance(role="host", party_id=config["host_id"]).component_param(
        with_label=False, output_format="dense"
    )

    # Intersection component
    intersect = Intersection(name="intersect")

    # HeteroNN component
    hetero_nn_param = HeteroNNParam(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        callback_param=callback_params,
        guest_bottom_model=GuestBottomModel(
            input_shape=config["guest_bottom_model"]["input_shape"],
            layers=config["guest_bottom_model"]["layers"]
        ),
        host_bottom_model=HostBottomModel(
            input_shape=config["host_bottom_model"]["input_shape"],
            layers=config["host_bottom_model"]["layers"]
        ),
        interactive_layer=InteractiveLayer(
            layer_type=config["interactive_layer"]["layer_type"],
            units=config["interactive_layer"]["units"]
        ),
        guest_top_model=GuestTopModel(
            layers=config["guest_top_model"]["layers"]
        )
    )
    hetero_nn = HeteroNN(name="hetero_nn", param=hetero_nn_param)

    # Evaluation component
    evaluation = Evaluation(name="evaluation", eval_type="binary")

    # Add components to pipeline
    pipeline.add_component(reader_guest)
    pipeline.add_component(reader_host)
    pipeline.add_component(data_transform_guest, data=reader_guest.output.data)
    pipeline.add_component(data_transform_host, data=reader_host.output.data)
    pipeline.add_component(intersect, data=[data_transform_guest.output.data, data_transform_host.output.data])
    pipeline.add_component(hetero_nn, data=intersect.output.data)
    pipeline.add_component(evaluation, data=hetero_nn.output.data)

    # Compile and fit pipeline
    pipeline.compile()
    pipeline.fit()

    # Print HeteroNN summary
    summary = pipeline.get_component("hetero_nn").get_summary()
    print("HeteroNN Summary:")
    print(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FATE HeteroNN Pipeline for Binary Classification")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    main(args.config)
```