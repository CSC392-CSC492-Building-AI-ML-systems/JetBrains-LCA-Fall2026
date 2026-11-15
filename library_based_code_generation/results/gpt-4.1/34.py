```python
import argparse
import sys
from fate_flow.pipeline import FateFlowPipeline
from fate_flow.pipeline.components.federated import (
    Reader,
    DataTransform,
    Intersection,
    FeatureScale,
    FeatureBinning,
    DataStatistics,
    Pearson,
    OneHotEncoder,
    FeatureSelection,
    LogisticRegression,
    Evaluation
)
from fate_flow.pipeline.interface import Data
from fate_flow.pipeline.utils.tools import load_job_config

def main(config_path):
    # Load config
    config = load_job_config(config_path)
    guest = config['role']['guest'][0]
    host = config['role']['host'][0]
    arbiter = config['role']['arbiter'][0]

    # Feature binning parameters
    binning_param = {
        "method": "quantile",
        "bin_num": 10,
        "adjustment_factor": 0.5,
        "transform_param": {"transform_type": "bin_num"}
    }

    # Feature selection parameters
    selection_param = {
        "select_col_indexes": -1,
        "filter_methods": ["iv_filter", "variance_filter"],
        "iv_param": {"metrics": "iv", "threshold": 0.1},
        "variance_param": {"threshold": 1e-5}
    }

    # Logistic regression parameters
    lr_param = {
        "penalty": "L2",
        "optimizer": "sgd",
        "max_iter": 10,
        "learning_rate": 0.15,
        "batch_size": 64,
        "init_param": {"init_method": "random_uniform"},
        "early_stop": "diff",
        "tol": 1e-4
    }

    # Reader
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(
        table=config['data']['guest_train_data'])
    reader_0.get_party_instance(role='host', party_id=host).component_param(
        table=config['data']['host_train_data'])

    # DataTransform
    data_transform_0 = DataTransform(name="data_transform_0", with_label=True, output_format="dense")
    data_transform_0.get_party_instance(role='guest', party_id=guest).component_param(
        label_name=config['data'].get('label_name', 'y'),
        label_type=config['data'].get('label_type', 'int'))
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(
        with_label=False)

    # Intersection
    intersection_0 = Intersection(name="intersection_0")

    # FeatureScale
    feature_scale_0 = FeatureScale(name="feature_scale_0", method="standard_scale")
    feature_scale_0.get_party_instance(role='guest', party_id=guest)
    feature_scale_0.get_party_instance(role='host', party_id=host)

    # FeatureBinning
    feature_binning_0 = FeatureBinning(name="feature_binning_0", **binning_param)

    # DataStatistics
    data_statistics_0 = DataStatistics(name="data_statistics_0")

    # Pearson
    pearson_0 = Pearson(name="pearson_0")

    # OneHotEncoder
    one_hot_encoder_0 = OneHotEncoder(name="one_hot_encoder_0")

    # FeatureSelection
    feature_selection_0 = FeatureSelection(name="feature_selection_0", **selection_param)

    # LogisticRegression
    lr_0 = LogisticRegression(name="lr_0", **lr_param)

    # Evaluation
    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    # Build pipeline
    pipeline = FateFlowPipeline()
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
    pipeline.add_component(lr_0, data=Data(train_data=feature_selection_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=lr_0.output.data))

    pipeline.compile()
    pipeline.fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FATE ML Pipeline")
    parser.add_argument('--config', type=str, default='examples/pipeline/config.yaml', help='Config file path')
    args = parser.parse_args()
    main(args.config)
```