```python
import argparse
import yaml
from fate_flow.pipeline import Pipeline
from fate_flow.pipeline.components.federated import (
    Reader, DataTransform, Intersection, FeatureScale, FeatureBinning, DataStatistics,
    Pearson, OneHotEncoder, FeatureSelection, HeteroLR, Evaluation
)
from fate_flow.pipeline.interface import Data
from fate_flow.pipeline.utils.tools import load_job_config

def main(config):
    guest = config['role']['guest'][0]
    host = config['role']['host'][0]
    arbiter = config['role']['arbiter'][0]

    # Reader
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=config['data']['guest'])
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=config['data']['host'])

    # DataTransform
    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role='guest', party_id=guest).component_param(with_label=True, output_format="dense")
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(with_label=False)

    # Intersection
    intersection_0 = Intersection(name="intersection_0")

    # FeatureScale
    feature_scale_0 = FeatureScale(name="feature_scale_0")
    feature_scale_0.get_party_instance(role='guest', party_id=guest).component_param(method="standard_scale")
    feature_scale_0.get_party_instance(role='host', party_id=host).component_param(method="standard_scale")

    # FeatureBinning
    feature_binning_param = {
        "method": "quantile",
        "bin_num": 10,
        "transform_param": {"transform_type": "bin_num"}
    }
    feature_binning_0 = FeatureBinning(name="feature_binning_0", **feature_binning_param)

    # DataStatistics
    data_statistics_0 = DataStatistics(name="data_statistics_0")

    # Pearson
    pearson_0 = Pearson(name="pearson_0")

    # OneHotEncoder
    one_hot_encoder_0 = OneHotEncoder(name="one_hot_encoder_0")

    # FeatureSelection
    feature_selection_param = {
        "select_col_indexes": -1,
        "filter_methods": ["iv_filter", "variance_filter", "manually"],
        "iv_param": {"threshold": 0.1},
        "variance_param": {"threshold": 1e-5}
    }
    feature_selection_0 = FeatureSelection(name="feature_selection_0", **feature_selection_param)

    # Logistic Regression
    lr_param = {
        "penalty": "L2",
        "optimizer": "sgd",
        "max_iter": 10,
        "learning_rate": 0.15,
        "init_param": {"init_method": "random_uniform"},
        "batch_size": -1,
        "early_stop": "diff"
    }
    hetero_lr_0 = HeteroLR(name="hetero_lr_0", **lr_param)

    # Evaluation
    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    # Pipeline
    pipeline = Pipeline()
    pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(feature_scale_0, data=Data(data=intersection_0.output.data))
    pipeline.add_component(feature_binning_0, data=Data(data=feature_scale_0.output.data))
    pipeline.add_component(data_statistics_0, data=Data(data=feature_binning_0.output.data))
    pipeline.add_component(pearson_0, data=Data(data=feature_binning_0.output.data))
    pipeline.add_component(one_hot_encoder_0, data=Data(data=feature_binning_0.output.data))
    pipeline.add_component(feature_selection_0, data=Data(data=one_hot_encoder_0.output.data))
    pipeline.add_component(hetero_lr_0, data=Data(train_data=feature_selection_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_lr_0.output.data))

    pipeline.compile()
    pipeline.fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FATE ML Pipeline")
    parser.add_argument("--config", "-c", type=str, default="default_config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config)
```