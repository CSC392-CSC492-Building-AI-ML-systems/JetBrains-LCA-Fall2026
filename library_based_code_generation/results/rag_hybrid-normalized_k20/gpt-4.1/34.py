```python
import sys
import argparse
import yaml
from fate.ml.pipeline import Pipeline
from fate.ml.components import (
    Reader,
    DataTransform,
    Intersection,
    FeatureScale,
    HeteroFeatureBinning,
    DataStatistics,
    Pearson,
    OneHotEncoder,
    HeteroFeatureSelection,
    HeteroLogisticRegression,
    Evaluation
)
from fate.ml.runtime import Roles

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="FATE ML Pipeline Script")
    parser.add_argument('--config', type=str, default='default_config.yaml', help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config)

    guest = config['role']['guest'][0]
    host = config['role']['host'][0]
    arbiter = config['role']['arbiter'][0]

    roles = Roles(guest=guest, host=host, arbiter=arbiter)

    # Feature binning parameters
    feature_binning_param = {
        "method": "quantile",
        "bin_num": 10,
        "adjustment_factor": 0.5,
        "local_only": False
    }

    # Feature selection parameters
    feature_selection_param = {
        "select_col_indexes": -1,
        "select_names": [],
        "filter_methods": ["iv_filter", "psi_filter"],
        "iv_param": {"threshold": 0.1},
        "psi_param": {"threshold": 0.1}
    }

    # Logistic regression parameters
    lr_param = {
        "penalty": "L2",
        "optimizer": "sgd",
        "max_iter": 10,
        "learning_rate": 0.15,
        "batch_size": 320,
        "early_stop": "diff",
        "init_param": {"init_method": "zeros"}
    }

    pipeline = Pipeline().set_roles(guest=guest, host=host, arbiter=arbiter)

    # Reader
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(
        table=config['data']['guest_train'])
    reader_0.get_party_instance(role='host', party_id=host).component_param(
        table=config['data']['host_train'])

    # DataTransform
    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role='guest', party_id=guest).component_param(
        with_label=True, output_format="dense")
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(
        with_label=False)

    # Intersection
    intersection_0 = Intersection(name="intersection_0")

    # FeatureScale
    feature_scale_0 = FeatureScale(name="feature_scale_0")
    feature_scale_0.get_party_instance(role='guest', party_id=guest).component_param(
        method="standard_scale")
    feature_scale_0.get_party_instance(role='host', party_id=host).component_param(
        method="standard_scale")

    # FeatureBinning
    feature_binning_0 = HeteroFeatureBinning(name="feature_binning_0", **feature_binning_param)

    # DataStatistics
    data_statistics_0 = DataStatistics(name="data_statistics_0")

    # Pearson
    pearson_0 = Pearson(name="pearson_0")

    # OneHotEncoder
    one_hot_encoder_0 = OneHotEncoder(name="one_hot_encoder_0")

    # FeatureSelection
    feature_selection_0 = HeteroFeatureSelection(name="feature_selection_0", **feature_selection_param)

    # LogisticRegression
    hetero_lr_0 = HeteroLogisticRegression(name="hetero_lr_0", **lr_param)

    # Evaluation
    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    # Add components to pipeline
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=reader_0.output.data)
    pipeline.add_component(intersection_0, data=data_transform_0.output.data)
    pipeline.add_component(feature_scale_0, data=intersection_0.output.data)
    pipeline.add_component(feature_binning_0, data=feature_scale_0.output.data)
    pipeline.add_component(data_statistics_0, data=feature_binning_0.output.data)
    pipeline.add_component(pearson_0, data=feature_binning_0.output.data)
    pipeline.add_component(one_hot_encoder_0, data=feature_binning_0.output.data)
    pipeline.add_component(feature_selection_0, data=one_hot_encoder_0.output.data)
    pipeline.add_component(hetero_lr_0, data=feature_selection_0.output.data)
    pipeline.add_component(evaluation_0, data=hetero_lr_0.output.data)

    # Compile and fit pipeline
    pipeline.compile()
    pipeline.fit()

if __name__ == "__main__":
    main()
```