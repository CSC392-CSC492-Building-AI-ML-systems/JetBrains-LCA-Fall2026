```python
import argparse
from fate.ml.pipeline import PipelineJob
from fate.ml.pipeline import PipelineConfig
from fate.ml.components import (
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

def main(config_path):
    # Load pipeline config
    config = PipelineConfig.load(config_path)
    guest = config.parties.guest[0]
    host = config.parties.host[0]
    guest_table = config.data.guest_train
    host_table = config.data.host_train

    # Define pipeline
    pipeline = PipelineJob(name="federated_learning_pipeline", config=config)

    # Reader
    reader_0 = Reader(name="reader_0")
    reader_0.component_param(
        table={"name": guest_table["name"], "namespace": guest_table["namespace"]},
        role="guest"
    )
    reader_0.component_param(
        table={"name": host_table["name"], "namespace": host_table["namespace"]},
        role="host"
    )
    pipeline.add_component(reader_0)

    # Data Transform
    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.component_param(
        with_label=True,
        output_format="dense"
    )
    pipeline.add_component(data_transform_0, data=reader_0.outputs["data"])

    # Sample
    sample_0 = Sample(name="sample_0")
    sample_0.component_param(
        mode="random",
        fraction=0.8,
        random_state=42
    )
    pipeline.add_component(sample_0, data=data_transform_0.outputs["data"])

    # Feature Binning
    binning_0 = FeatureBinning(name="binning_0")
    binning_0.component_param(
        method="quantile",
        bin_num=10
    )
    pipeline.add_component(binning_0, data=sample_0.outputs["data"])

    # One Hot Encoder
    one_hot_0 = OneHotEncoder(name="one_hot_0")
    one_hot_0.component_param()
    pipeline.add_component(one_hot_0, data=binning_0.outputs["data"])

    # Logistic Regression
    lr_0 = LogisticRegression(name="lr_0")
    lr_0.component_param(
        penalty="L2",
        max_iter=10,
        alpha=0.01,
        optimizer="sgd"
    )
    pipeline.add_component(lr_0, data=one_hot_0.outputs["data"])

    # Local Baseline
    baseline_0 = LocalBaseline(name="baseline_0")
    baseline_0.component_param()
    pipeline.add_component(baseline_0, data=one_hot_0.outputs["data"])

    # Secure Boost
    sbt_0 = SecureBoost(name="sbt_0")
    sbt_0.component_param(
        num_trees=5,
        max_depth=3,
        learning_rate=0.1,
        objective="binary:logistic"
    )
    pipeline.add_component(sbt_0, data=one_hot_0.outputs["data"])

    # Evaluation
    eval_lr = Evaluation(name="eval_lr")
    eval_lr.component_param(
        eval_type="binary"
    )
    pipeline.add_component(eval_lr, data=lr_0.outputs["predict"])

    eval_sbt = Evaluation(name="eval_sbt")
    eval_sbt.component_param(
        eval_type="binary"
    )
    pipeline.add_component(eval_sbt, data=sbt_0.outputs["predict"])

    # Compile and fit pipeline
    pipeline.compile()
    pipeline.fit()

    # Print evaluation summaries
    print("Logistic Regression Evaluation Summary:")
    print(eval_lr.get_summary())
    print("Secure Boost Evaluation Summary:")
    print(eval_sbt.get_summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Pipeline with FATE")
    parser.add_argument("--config", type=str, required=True, help="Path to pipeline config file")
    args = parser.parse_args()
    main(args.config)
```