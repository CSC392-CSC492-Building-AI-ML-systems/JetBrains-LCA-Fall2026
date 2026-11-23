```python
import argparse
import json
from fate_flow.pipeline import FateFlowPipeline
from fate_flow.components import Reader, DataTransform, FeatureScale, LogisticRegression, Evaluation
from fate_flow.entity import Role, JobParameters, Table
from fate_flow.utils import load_job_config

def main(config_path):
    # Load configuration
    config = load_job_config(config_path)
    guest = config['role']['guest'][0]
    host = config['role']['host'][0]
    arbiter = config['role']['arbiter'][0]
    guest_table = config['guest_table']
    host_table = config['host_table']

    # Define roles
    roles = {
        'guest': [guest],
        'host': [host],
        'arbiter': [arbiter]
    }

    # Initialize pipeline
    pipeline = FateFlowPipeline()
    pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)

    # Reader for guest
    reader_0 = Reader(name='reader_0')
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(
        table=Table(name=guest_table['name'], namespace=guest_table['namespace'])
    )
    # Reader for host
    reader_0.get_party_instance(role='host', party_id=host).component_param(
        table=Table(name=host_table['name'], namespace=host_table['namespace'])
    )

    # DataTransform
    data_transform_0 = DataTransform(name='data_transform_0', with_label=True, output_format='dense')
    data_transform_0.get_party_instance(role='guest', party_id=guest).component_param(label_name='y', label_type='int')
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(with_label=False)

    # FeatureScale
    scale_0 = FeatureScale(name='scale_0', method='standard_scale')

    # Logistic Regression
    lr_0 = LogisticRegression(
        name='lr_0',
        penalty='L2',
        optimizer='sgd',
        tol=1e-4,
        alpha=0.01,
        max_iter=30,
        early_stop='diff',
        batch_size=64,
        learning_rate=0.15,
        decay=0.01,
        init_method='random_uniform',
        cv_param={
            'n_splits': 5,
            'shuffle': True,
            'random_seed': 42,
            'need_cv': True
        }
    )

    # Evaluation
    eval_0 = Evaluation(name='eval_0', eval_type='binary')

    # Add components to pipeline
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=reader_0.output.data)
    pipeline.add_component(scale_0, data=data_transform_0.output.data)
    pipeline.add_component(lr_0, data=scale_0.output.data)
    pipeline.add_component(eval_0, data=lr_0.output.data)

    # Compile and fit pipeline
    pipeline.compile()
    pipeline.fit(JobParameters())

    # Deploy selected components
    pipeline.deploy_component([data_transform_0, scale_0, lr_0])

    # Prediction pipeline
    predict_pipeline = FateFlowPipeline()
    predict_pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)

    # Use the same reader for prediction
    reader_1 = Reader(name='reader_1')
    reader_1.get_party_instance(role='guest', party_id=guest).component_param(
        table=Table(name=guest_table['name'], namespace=guest_table['namespace'])
    )
    reader_1.get_party_instance(role='host', party_id=host).component_param(
        table=Table(name=host_table['name'], namespace=host_table['namespace'])
    )

    predict_pipeline.add_component(reader_1)
    predict_pipeline.add_component(
        pipeline.get_component('data_transform_0'),
        data=reader_1.output.data
    )
    predict_pipeline.add_component(
        pipeline.get_component('scale_0'),
        data=predict_pipeline.get_component('data_transform_0').output.data
    )
    predict_pipeline.add_component(
        pipeline.get_component('lr_0'),
        data=predict_pipeline.get_component('scale_0').output.data
    )

    # Compile and predict
    predict_pipeline.compile()
    predict_pipeline.predict(JobParameters())

    # Save DSL and config as JSON
    with open('predict_pipeline_dsl.json', 'w') as f:
        json.dump(predict_pipeline.get_dsl(), f, indent=4)
    with open('predict_pipeline_conf.json', 'w') as f:
        json.dump(predict_pipeline.get_conf(), f, indent=4)

    # Print summaries
    print("Logistic Regression Summary:")
    print(json.dumps(pipeline.get_component('lr_0').get_summary(), indent=4))
    print("Evaluation Summary:")
    print(json.dumps(pipeline.get_component('eval_0').get_summary(), indent=4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config)
```