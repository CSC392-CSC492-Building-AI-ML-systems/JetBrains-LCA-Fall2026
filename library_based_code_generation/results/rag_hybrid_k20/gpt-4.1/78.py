```python
import argparse
from fate.ml.pipeline import Pipeline
from fate.ml.component import (
    Reader,
    DataTransform,
    Sample,
    FeatureBinning,
    OneHotEncoder,
    HeteroLR,
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
    backend = config.backend
    work_mode = config.work_mode

    # Define data tables
    guest_train_data = {"name": config.dataset.guest_train.name, "namespace": config.dataset.guest_train.namespace}
    host_train_data = {"name": config.dataset.host_train.name, "namespace": config.dataset.host_train.namespace}

    # Pipeline initialization
    pipeline = Pipeline().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host)

    # Reader
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    # Data Transform
    data_transform_0 = DataTransform(name="data_transform_0", with_label=True, output_format="dense")

    # Sampling
    sample_0 = Sample(name="sample_0", mode="random", fraction=0.8, random_state=42)

    # Feature Binning
    binning_0 = FeatureBinning(name="binning_0", method="quantile", compress_thres=10000, bin_num=10)

    # One Hot Encoding
    one_hot_0 = OneHotEncoder(name="one_hot_0")

    # Logistic Regression
    lr_0 = HeteroLR(name="hetero_lr_0", penalty="L2", optimizer="sgd", tol=1e-4, alpha=0.01, max_iter=10, batch_size=320, learning_rate=0.15, init_param={"init_method": "random_uniform"})

    # Local Baseline Model
    baseline_0 = LocalBaseline(name="local_baseline_0", model_type="lr")

    # Secure Boosting
    sbt_0 = SecureBoost(name="secure_boost_0", num_trees=5, task_type="classification", objective_param={"objective": "cross_entropy"}, learning_rate=0.1, tree_param={"max_depth": 3})

    # Evaluation
    eval_0 = Evaluation(name="eval_0", eval_type="binary")

    # Add components to pipeline
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=reader_0.output.data)
    pipeline.add_component(sample_0, data=data_transform_0.output.data)
    pipeline.add_component(binning_0, data=sample_0.output.data)
    pipeline.add_component(one_hot_0, data=binning_0.output.data)
    pipeline.add_component(lr_0, data=one_hot_0.output.data)
    pipeline.add_component(baseline_0, data=one_hot_0.output.data)
    pipeline.add_component(sbt_0, data=one_hot_0.output.data)
    pipeline.add_component(eval_0, data=[lr_0.output.data, sbt_0.output.data, baseline_0.output.data])

    # Compile and fit pipeline
    pipeline.compile()
    pipeline.fit(backend=backend, work_mode=work_mode)

    # Print evaluation summary
    print("Evaluation Summary:")
    print(eval_0.get_summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Pipeline with FATE")
    parser.add_argument("--config", type=str, required=True, help="Path to pipeline config file")
    args = parser.parse_args()
    main(args.config)
```