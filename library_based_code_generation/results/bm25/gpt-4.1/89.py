```python
import argparse
import json
from fate.ml.pipeline import Pipeline, predict_pipeline
from fate.ml.components import Reader, DataTransform, FeatureScale, LogisticRegression, Evaluation
from fate.ml.pipeline import PipelineConfig

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    guest = config['guest']
    host = config['host']
    arbiter = config['arbiter']
    guest_table = config['guest_table']
    host_table = config['host_table']

    # Initialize pipeline
    pipeline = Pipeline()
    pipeline.set_initiator(role='guest', party_id=guest)
    pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)

    # Reader for guest
    reader_0 = Reader(name='reader_0')
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(
        table=guest_table
    )
    reader_0.get_party_instance(role='host', party_id=host).component_param(
        table=host_table
    )

    # Data Transform
    data_transform_0 = DataTransform(name='data_transform_0')
    data_transform_0.get_party_instance(role='guest', party_id=guest).component_param(
        with_label=True,
        output_format='dense'
    )
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(
        with_label=False
    )

    # Feature Scale
    scale_0 = FeatureScale(name='scale_0')
    scale_0.get_party_instance(role='guest', party_id=guest).component_param(
        method='standard_scale'
    )
    scale_0.get_party_instance(role='host', party_id=host).component_param(
        method='standard_scale'
    )

    # Logistic Regression
    lr_0 = LogisticRegression(name='lr_0')
    lr_0.get_party_instance(role='guest', party_id=guest).component_param(
        penalty='L2',
        optimizer='sgd',
        tol=1e-4,
        alpha=0.01,
        max_iter=50,
        early_stop='diff',
        batch_size=64,
        learning_rate=0.05,
        decay=0.001,
        init_method='random_uniform',
        cv_param={
            'n_splits': 5,
            'shuffle': True,
            'random_seed': 42,
            'need_cv': True
        }
    )

    # Evaluation
    eval_0 = Evaluation(name='eval_0')
    eval_0.get_party_instance(role='guest', party_id=guest).component_param(
        eval_type='binary'
    )

    # Add components to pipeline
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=reader_0.output.data)
    pipeline.add_component(scale_0, data=data_transform_0.output.data)
    pipeline.add_component(lr_0, data=scale_0.output.data)
    pipeline.add_component(eval_0, data=lr_0.output.data)

    # Compile and fit pipeline
    pipeline.compile()
    pipeline.fit()

    # Deploy selected components
    pipeline.deploy_component([data_transform_0, scale_0, lr_0])

    # Prediction pipeline
    predict_pipe = predict_pipeline.PredictPipeline()
    predict_pipe.add_component(reader_0, data=reader_0.output.data)
    predict_pipe.add_component(data_transform_0, data=reader_0.output.data)
    predict_pipe.add_component(scale_0, data=data_transform_0.output.data)
    predict_pipe.add_component(lr_0, data=scale_0.output.data)

    # Compile prediction pipeline
    predict_pipe.compile()

    # Make predictions
    predict_job = predict_pipe.fit()

    # Save DSL and config as JSON
    with open('predict_pipeline_dsl.json', 'w') as f:
        json.dump(predict_pipe.get_dsl(), f, indent=4)
    with open('predict_pipeline_conf.json', 'w') as f:
        json.dump(predict_pipe.get_conf(), f, indent=4)

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