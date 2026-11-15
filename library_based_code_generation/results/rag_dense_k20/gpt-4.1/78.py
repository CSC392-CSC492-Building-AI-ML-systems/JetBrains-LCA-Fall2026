```python
import argparse
from fate_flow.pipeline import Pipeline
from fate_flow.components import Reader, DataTransform, FederatedSample, FeatureBinning, OneHotEncoder, HeteroLR, LocalBaseline, HeteroSecureBoost, Evaluation
from fate_flow.entity import Guest, Host, Arbiter
from fate_flow.utils import load_job_config

def main(config_path):
    # Load config
    config = load_job_config(config_path)
    guest_id = config['guest']
    host_id = config['host']
    arbiter_id = config.get('arbiter', None)
    guest_table = config['guest_data']['name'], config['guest_data']['namespace']
    host_table = config['host_data']['name'], config['host_data']['namespace']

    # Parties
    guest = Guest(guest_id)
    host = Host(host_id)
    if arbiter_id:
        arbiter = Arbiter(arbiter_id)
        parties = [guest, host, arbiter]
    else:
        parties = [guest, host]

    # Pipeline
    pipeline = Pipeline().set_initiator(role='guest', party_id=guest_id).set_roles(guest=guest_id, host=host_id, arbiter=arbiter_id)

    # Reader
    reader_0 = Reader(name='reader_0')
    reader_0.get_party_instance(role='guest', party_id=guest_id).component_param(table=guest_table)
    reader_0.get_party_instance(role='host', party_id=host_id).component_param(table=host_table)

    # DataTransform
    data_transform_0 = DataTransform(name='data_transform_0', with_label=True, output_format='dense')
    data_transform_0.get_party_instance(role='guest', party_id=guest_id).component_param(label_name='y', label_type='int')
    data_transform_0.get_party_instance(role='host', party_id=host_id).component_param(with_label=False)

    # FederatedSample
    federated_sample_0 = FederatedSample(name='federated_sample_0', mode='random', fraction=0.8, random_state=42)

    # FeatureBinning
    feature_binning_0 = FeatureBinning(name='feature_binning_0', method='quantile', bin_num=10)

    # OneHotEncoder
    one_hot_encoder_0 = OneHotEncoder(name='one_hot_encoder_0')

    # Logistic Regression
    hetero_lr_0 = HeteroLR(name='hetero_lr_0', penalty='L2', optimizer='sgd', max_iter=10, alpha=0.01, batch_size=64, learning_rate=0.15, init_param={'init_method': 'random_uniform'})

    # Local Baseline
    local_baseline_0 = LocalBaseline(name='local_baseline_0', model_type='lr')

    # Secure Boosting
    hetero_secureboost_0 = HeteroSecureBoost(name='hetero_secureboost_0', num_trees=5, task_type='classification', objective_param={'objective': 'cross_entropy'}, encrypt_param={'method': 'Paillier'})

    # Evaluation
    evaluation_0 = Evaluation(name='evaluation_0', eval_type='binary')

    # Add components to pipeline
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=reader_0.output.data)
    pipeline.add_component(federated_sample_0, data=data_transform_0.output.data)
    pipeline.add_component(feature_binning_0, data=federated_sample_0.output.data)
    pipeline.add_component(one_hot_encoder_0, data=feature_binning_0.output.data)
    pipeline.add_component(hetero_lr_0, data=one_hot_encoder_0.output.data)
    pipeline.add_component(local_baseline_0, data=one_hot_encoder_0.output.data)
    pipeline.add_component(hetero_secureboost_0, data=one_hot_encoder_0.output.data)
    pipeline.add_component(evaluation_0, data=[hetero_lr_0.output.data, local_baseline_0.output.data, hetero_secureboost_0.output.data])

    # Compile and fit
    pipeline.compile()
    pipeline.fit()

    # Print evaluation summary
    print("Evaluation Summary:")
    print(evaluation_0.get_summary())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Federated Learning Pipeline with FATE")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config)
```