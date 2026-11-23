```python
import argparse
from fate.arch import Context
from fate.ml.nn.hetero_nn import HeteroNN, HeteroNNParam, HeteroNNGuestModel, HeteroNNHostModel
from fate.ml.nn.hetero_nn import HeteroNNGuest, HeteroNNHost
from fate.ml.nn.hetero_nn import set_hetero_nn_model_meta, set_hetero_nn_model_param
from fate.ml.pipeline import Pipeline
from fate.ml.components import Reader, DataTransform, Intersection, Evaluation

def main(config_path):
    # Load config
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    guest = config["role"]["guest"][0]
    host = config["role"]["host"][0]
    backend = config["backend"]
    work_mode = config["work_mode"]

    # Define pipeline
    pipeline = Pipeline().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host)

    # Reader component
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=config["data"]["guest_train"])
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=config["data"]["host_train"])

    # DataTransform component
    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role='guest', party_id=guest).component_param(with_label=True, output_format="dense")
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(with_label=False)

    # Intersection component
    intersect_0 = Intersection(name="intersect_0")

    # HeteroNN component
    hetero_nn_param = HeteroNNParam(
        epochs=5,
        batch_size=64,
        learning_rate=0.01,
        optimizer="SGD",
        guest_bottom_model=[
            {"layer": "Dense", "units": 64, "activation": "relu"},
            {"layer": "Dense", "units": 32, "activation": "relu"}
        ],
        host_bottom_model=[
            {"layer": "Dense", "units": 64, "activation": "relu"},
            {"layer": "Dense", "units": 32, "activation": "relu"}
        ],
        interactive_layer={"layer": "InteractiveLayer", "units": 32, "activation": "relu"},
        guest_top_model=[
            {"layer": "Dense", "units": 1, "activation": "sigmoid"}
        ],
        config_type="keras",
        callback_param={
            "early_stopping": True,
            "patience": 2,
            "save_best_only": True
        }
    )
    hetero_nn_0 = HeteroNN(name="hetero_nn_0", param=hetero_nn_param)

    # Evaluation component
    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    # Add components to pipeline
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=reader_0.output.data)
    pipeline.add_component(intersect_0, data=data_transform_0.output.data)
    pipeline.add_component(hetero_nn_0, data=intersect_0.output.data)
    pipeline.add_component(evaluation_0, data=hetero_nn_0.output.data)

    # Compile and fit pipeline
    pipeline.compile()
    pipeline.fit(backend=backend, work_mode=work_mode)

    # Print HeteroNN summary
    summary = pipeline.get_component("hetero_nn_0").get_summary()
    print("HeteroNN Summary:")
    print(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FATE HeteroNN Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)
```