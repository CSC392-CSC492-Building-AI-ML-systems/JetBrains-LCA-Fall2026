```python
import argparse
from fate.ml.pipeline import Pipeline
from fate.ml.component import (
    Reader,
    DataTransform,
    Sample,
    FeatureBinning,
    OneHotEncoder,
    LogisticRegression,
    LocalBaseline,
    SecureBoost,
    Evaluation
)
from fate.ml.runtime import PipelineConfig

def main(config_path):
    # Load pipeline config
    config = PipelineConfig.load(config_path)
    guest = config.parties.guest[0]
    host = config.parties.host[0]

    # Define data tables
    guest_train_data = {"name": config.data.guest_train.name, "namespace": config.data.guest_train.namespace}
    host_train_data = {"name": config.data.host_train.name, "namespace": config.data.host_train.namespace}

    # Reader
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role="guest", party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role="host", party_id=host).component_param(table=host_train_data)

    # Data Transform
    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role="guest", party_id=guest).component_param(with_label=True, output_format="dense")
    data_transform_0.get_party_instance(role="host", party_id=host).component_param(with_label=False)

    # Sample
    sample_0 = Sample(name="sample_0")
    sample_0.get_party_instance(role="guest", party_id=guest).component_param(fraction=0.8, random_state=42)
    sample_0.get_party_instance(role="host", party_id=host).component_param(fraction=0.8, random_state=42)

    # Feature Binning
    binning_0 = FeatureBinning(name="binning_0")
    binning_0.get_party_instance(role="guest", party_id=guest).component_param(method="quantile", compress_thres=10000, bin_num=10)
    binning_0.get_party_instance(role="host", party_id=host).component_param(method="quantile", compress_thres=10000, bin_num=10)

    # One Hot Encoder
    one_hot_0 = OneHotEncoder(name="one_hot_0")
    one_hot_0.get_party_instance(role="guest", party_id=guest).component_param()
    one_hot_0.get_party_instance(role="host", party_id=host).component_param()

    # Logistic Regression
    lr_0 = LogisticRegression(name="lr_0")
    lr_0.get_party_instance(role="guest", party_id=guest).component_param(penalty="L2", max_iter=10, alpha=0.01, optimizer="sgd", batch_size=64, learning_rate=0.15, init_method="random_uniform")
    lr_0.get_party_instance(role="host", party_id=host).component_param()

    # Local Baseline
    baseline_0 = LocalBaseline(name="baseline_0")
    baseline_0.get_party_instance(role="guest", party_id=guest).component_param(model_type="lr")
    baseline_0.get_party_instance(role="host", party_id=host).component_param(model_type="lr")

    # Secure Boost
    sbt_0 = SecureBoost(name="sbt_0")
    sbt_0.get_party_instance(role="guest", party_id=guest).component_param(num_trees=5, task_type="classification", objective_param={"objective": "cross_entropy"}, learning_rate=0.1, max_depth=3)
    sbt_0.get_party_instance(role="host", party_id=host).component_param()

    # Evaluation
    eval_0 = Evaluation(name="eval_0")
    eval_0.get_party_instance(role="guest", party_id=guest).component_param(eval_type="binary")
    eval_0.get_party_instance(role="host", party_id=host).component_param(eval_type="binary")

    # Build pipeline
    pipeline = Pipeline()
    pipeline.set_initiator(role="guest", party_id=guest)
    pipeline.set_roles(guest=guest, host=host)

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=reader_0.output.data)
    pipeline.add_component(sample_0, data=data_transform_0.output.data)
    pipeline.add_component(binning_0, data=sample_0.output.data)
    pipeline.add_component(one_hot_0, data=binning_0.output.data)
    pipeline.add_component(lr_0, data=one_hot_0.output.data)
    pipeline.add_component(baseline_0, data=one_hot_0.output.data)
    pipeline.add_component(sbt_0, data=one_hot_0.output.data)
    pipeline.add_component(eval_0, data=[lr_0.output.data, baseline_0.output.data, sbt_0.output.data])

    # Compile and fit
    pipeline.compile()
    pipeline.fit()

    # Print evaluation summary
    print("Evaluation Summary:")
    print(eval_0.get_summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Pipeline with FATE")
    parser.add_argument("--config", type=str, required=True, help="Path to pipeline config file")
    args = parser.parse_args()
    main(args.config)
```