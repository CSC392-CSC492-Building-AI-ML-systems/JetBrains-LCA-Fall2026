```python
import argparse
from fate_flow.pipeline import FateFlowPipeline
from fate_flow.components import Reader, DataTransform, Sample, FeatureBinning, OneHotEncoder, LogisticRegression, LocalBaseline, SecureBoost, Evaluation
from fate_flow.entity import Role, Party

def main(config_path):
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)

    guest_id = config['guest']
    host_id = config['host']
    guest_table = config['guest_table']
    host_table = config['host_table']

    pipeline = FateFlowPipeline()

    pipeline.set_roles(
        guest=Party(guest_id),
        host=Party(host_id)
    )

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(Party(guest_id)).component_param(table=guest_table)
    reader_0.get_party_instance(Party(host_id)).component_param(table=host_table)

    data_transform_0 = DataTransform(name="data_transform_0", with_label=True, output_format="dense")
    sample_0 = Sample(name="sample_0", mode="random", fraction=0.8)
    feature_binning_0 = FeatureBinning(name="feature_binning_0", method="quantile", bin_num=10)
    one_hot_encoder_0 = OneHotEncoder(name="one_hot_encoder_0")
    lr_0 = LogisticRegression(name="lr_0", penalty="L2", max_iter=10, alpha=0.01)
    local_baseline_0 = LocalBaseline(name="local_baseline_0")
    secure_boost_0 = SecureBoost(name="secure_boost_0", num_trees=5, task_type="classification")
    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=reader_0.output.data)
    pipeline.add_component(sample_0, data=data_transform_0.output.data)
    pipeline.add_component(feature_binning_0, data=sample_0.output.data)
    pipeline.add_component(one_hot_encoder_0, data=feature_binning_0.output.data)
    pipeline.add_component(lr_0, data=one_hot_encoder_0.output.data)
    pipeline.add_component(local_baseline_0, data=one_hot_encoder_0.output.data)
    pipeline.add_component(secure_boost_0, data=one_hot_encoder_0.output.data)
    pipeline.add_component(evaluation_0, data=[lr_0.output.data, local_baseline_0.output.data, secure_boost_0.output.data])

    pipeline.compile()
    pipeline.fit()

    print("Evaluation Summary:")
    print(evaluation_0.get_summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Pipeline with FATE")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()
    main(args.config)
```