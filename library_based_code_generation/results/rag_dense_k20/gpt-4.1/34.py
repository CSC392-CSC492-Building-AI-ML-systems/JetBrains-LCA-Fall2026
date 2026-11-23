```python
import sys
import argparse
from fate_flow.pipeline import Pipeline
from fate_flow.pipeline.component import Reader, DataTransform, Intersection, FeatureScale, FeatureBinning, DataStatistics, Pearson, OneHotEncoder, FeatureSelection, HeteroLR, Evaluation
from fate_flow.pipeline.interface import Data
from fate_flow.pipeline.runtime.entity import JobParameters

def get_config(config_path=None):
    import yaml
    default_config = {
        "guest": 10000,
        "host": 10001,
        "arbiter": 10000,
        "guest_data": {"name": "guest_data", "namespace": "experiment"},
        "host_data": {"name": "host_data", "namespace": "experiment"},
        "feature_binning_param": {
            "method": "quantile",
            "bin_num": 10
        },
        "feature_selection_param": {
            "select_col_indexes": -1,
            "select_names": [],
            "filter_methods": ["iv_filter"],
            "iv_param": {"threshold": 0.1}
        },
        "lr_param": {
            "penalty": "L2",
            "optimizer": "sgd",
            "alpha": 0.01,
            "max_iter": 10,
            "batch_size": 64,
            "learning_rate": 0.15,
            "init_param": {"init_method": "random_uniform"}
        }
    }
    if config_path:
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
        default_config.update(user_config)
    return default_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    config = get_config(args.config)

    guest = config["guest"]
    host = config["host"]
    arbiter = config["arbiter"]

    # Components
    reader_0 = Reader(name="reader_0")
    reader_1 = Reader(name="reader_1")
    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_1 = DataTransform(name="data_transform_1")
    intersection_0 = Intersection(name="intersection_0")
    feature_scale_0 = FeatureScale(name="feature_scale_0")
    feature_scale_1 = FeatureScale(name="feature_scale_1")
    feature_binning_0 = FeatureBinning(name="feature_binning_0", **config["feature_binning_param"])
    data_statistics_0 = DataStatistics(name="data_statistics_0")
    pearson_0 = Pearson(name="pearson_0")
    one_hot_encoder_0 = OneHotEncoder(name="one_hot_encoder_0")
    feature_selection_0 = FeatureSelection(name="feature_selection_0", **config["feature_selection_param"])
    hetero_lr_0 = HeteroLR(name="hetero_lr_0", **config["lr_param"])
    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    # Pipeline
    pipeline = Pipeline()
    pipeline.set_initiator(role='guest', party_id=guest)
    pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)

    # Reader
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(
        table=config["guest_data"]
    )
    reader_1.get_party_instance(role='host', party_id=host).component_param(
        table=config["host_data"]
    )

    # DataTransform
    data_transform_0.get_party_instance(role='guest', party_id=guest).component_param(
        with_label=True, output_format="dense"
    )
    data_transform_1.get_party_instance(role='host', party_id=host).component_param(
        with_label=False
    )

    # FeatureScale
    feature_scale_0.get_party_instance(role='guest', party_id=guest).component_param(
        method="standard_scale"
    )
    feature_scale_1.get_party_instance(role='host', party_id=host).component_param(
        method="standard_scale"
    )

    # Add components to pipeline
    pipeline.add_component(reader_0, data=Data(data={"data": "args"}))
    pipeline.add_component(reader_1, data=Data(data={"data": "args"}))
    pipeline.add_component(data_transform_0, data=Data(data={"data": reader_0.output.data}))
    pipeline.add_component(data_transform_1, data=Data(data={"data": reader_1.output.data}))
    pipeline.add_component(intersection_0, data=Data(data={"data": [data_transform_0.output.data, data_transform_1.output.data]}))
    pipeline.add_component(feature_scale_0, data=Data(data={"data": intersection_0.output.data}))
    pipeline.add_component(feature_scale_1, data=Data(data={"data": intersection_0.output.data}))
    pipeline.add_component(feature_binning_0, data=Data(data={"data": feature_scale_0.output.data}))
    pipeline.add_component(data_statistics_0, data=Data(data={"data": feature_binning_0.output.data}))
    pipeline.add_component(pearson_0, data=Data(data={"data": feature_binning_0.output.data}))
    pipeline.add_component(one_hot_encoder_0, data=Data(data={"data": feature_binning_0.output.data}))
    pipeline.add_component(feature_selection_0, data=Data(data={"data": one_hot_encoder_0.output.data}))
    pipeline.add_component(hetero_lr_0, data=Data(data={"train_data": feature_selection_0.output.data}))
    pipeline.add_component(evaluation_0, data=Data(data={"data": hetero_lr_0.output.data}))

    # Compile and fit
    pipeline.compile()
    pipeline.fit(JobParameters())

if __name__ == "__main__":
    main()
```