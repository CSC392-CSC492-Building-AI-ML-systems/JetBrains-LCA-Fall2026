```python
import argparse
import json
from fate.ml.pipeline import Pipeline
from fate.ml.components import Reader, DataTransform, FeatureScale, LogisticRegression, Evaluation
from fate.ml.runtime import get_party_instance
from fate.ml.pipeline.utils import deploy_component
from fate.ml.pipeline.predict_pipeline import PredictPipeline

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    guest = config['guest']
    host = config['host']
    arbiter = config['arbiter']
    guest_table = config['guest_table']
    host_table = config['host_table']

    # Step 1: Define pipeline
    pipeline = Pipeline()
    pipeline.set_initiator(role='guest', party_id=guest)
    pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)

    # Step 2: Reader
    reader_0 = Reader(name='reader_0')
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_table)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_table)

    # Step 3: Data Transform
    data_transform_0 = DataTransform(name='data_transform_0', with_label=True, output_format='dense')
    data_transform_0.get_party_instance(role='guest', party_id=guest).component_param(label_name='y')

    # Step 4: Feature Scale
    scale_0 = FeatureScale(name='scale_0', method='standard_scale', need_run=True)

    # Step 5: Logistic Regression
    lr_0 = LogisticRegression(
        name='lr_0',
        penalty='L2',
        optimizer='sgd',
        tol=1e-4,
        alpha=0.01,
        max_iter=30,
        early_stop='diff',
        batch_size=320,
        learning_rate=0.15,
        decay=0.0,
        init_method='random_uniform',
        cv_param={
            'n_splits': 5,
            'shuffle': True,
            'random_seed': 42,
            'need_cv': True
        }
    )

    # Step 6: Evaluation
    eval_0 = Evaluation(name='eval_0', eval_type='binary')

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
    deploy_component_list = [data_transform_0, scale_0, lr_0]
    pipeline.deploy_component(*deploy_component_list)

    # Prediction pipeline
    predict_pipeline = PredictPipeline()
    predict_pipeline.add_component(reader_0)
    predict_pipeline.add_component(data_transform_0, data=reader_0.output.data)
    predict_pipeline.add_component(scale_0, data=data_transform_0.output.data)
    predict_pipeline.add_component(lr_0, data=scale_0.output.data)

    # Compile and predict
    predict_pipeline.compile()
    predict_pipeline.predict()

    # Save DSL and config
    with open('predict_pipeline_dsl.json', 'w') as f:
        json.dump(predict_pipeline.get_dsl(), f, indent=4)
    with open('predict_pipeline_conf.json', 'w') as f:
        json.dump(predict_pipeline.get_conf(), f, indent=4)

    # Print summaries
    print("Logistic Regression Summary:")
    print(json.dumps(lr_0.get_summary(), indent=4))
    print("Evaluation Summary:")
    print(json.dumps(eval_0.get_summary(), indent=4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config)
```