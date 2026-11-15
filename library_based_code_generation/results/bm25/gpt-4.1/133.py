```python
import argparse
from fate.arch import Context
from fate.ml.pipeline import Pipeline
from fate.ml.component import Reader, DataTransform, Intersection, HeteroNN, Evaluation

def main(config_path):
    # Load configuration
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    guest_data = config['guest_data']
    host_data = config['host_data']
    guest = config['guest']
    host = config['host']

    # Reader components
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(
        table=guest_data
    )
    reader_0.get_party_instance(role='host', party_id=host).component_param(
        table=host_data
    )

    # DataTransform components
    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role='guest', party_id=guest).component_param(
        with_label=True,
        output_format="dense"
    )
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(
        with_label=False
    )

    # Intersection component
    intersect_0 = Intersection(name="intersection_0")

    # HeteroNN component
    hetero_nn_0 = HeteroNN(name="hetero_nn_0")
    hetero_nn_0.get_party_instance(role='guest', party_id=guest).component_param(
        config_type="keras",
        epochs=10,
        batch_size=64,
        learning_rate=0.01,
        optimizer="Adam",
        early_stop={"early_stop": "diff", "eps": 1e-4},
        guest_bottom_model={
            "layers": [
                {"units": 64, "activation": "relu"},
                {"units": 32, "activation": "relu"}
            ]
        },
        interactive_layer={
            "units": 16,
            "activation": "relu"
        },
        guest_top_model={
            "layers": [
                {"units": 1, "activation": "sigmoid"}
            ]
        },
        validation_freqs=1,
        callbacks=[
            {"early_stopping": {"monitor": "val_loss", "patience": 3}},
            {"model_checkpoint": {"save_best_only": True}}
        ]
    )
    hetero_nn_0.get_party_instance(role='host', party_id=host).component_param(
        host_bottom_model={
            "layers": [
                {"units": 64, "activation": "relu"},
                {"units": 32, "activation": "relu"}
            ]
        }
    )

    # Evaluation component
    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    # Build pipeline
    pipeline = Pipeline()
    pipeline.set_roles(guest=guest, host=host)
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=reader_0.output.data)
    pipeline.add_component(intersect_0, data=data_transform_0.output.data)
    pipeline.add_component(hetero_nn_0, data=intersect_0.output.data)
    pipeline.add_component(evaluation_0, data=hetero_nn_0.output.data)

    # Compile and fit pipeline
    pipeline.compile()
    pipeline.fit()

    # Print HeteroNN summary
    summary = pipeline.get_component("hetero_nn_0").get_summary()
    print("HeteroNN Summary:", summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FATE HeteroNN Binary Classification Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    main(args.config)
```