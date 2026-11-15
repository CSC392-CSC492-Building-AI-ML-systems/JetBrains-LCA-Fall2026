```python
import argparse
import json
from fate.ml.pipeline import Pipeline, predict_pipeline
from fate.ml.components import Reader, DataTransform, FeatureScale, LogisticRegression, Evaluation
from fate.ml.runtime import PipelineConfig

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    guest = config['role']['guest'][0]
    host = config['role']['host'][0]
    arbiter = config['role']['arbiter'][0]
    guest_table = config['data']['guest_train']
    host_table = config['data']['host_train']

    # Step 1: Reader
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_table)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_table)

    # Step 2: Data Transform
    data_transform_0 = DataTransform(name="data_transform_0", with_label=True, output_format="dense")
    data_transform_0.get_party_instance(role='guest', party_id=guest).component_param(label_name=config['data']['label_name'])

    # Step 3: Feature Scale
    scale_0 = FeatureScale(name="scale_0", method="standard_scale", need_run=True)

    # Step 4: Logistic Regression
    lr_0 = LogisticRegression(
        name="lr_0",
        penalty=config['lr']['penalty'],
        optimizer=config['lr']['optimizer'],
        tol=config['lr']['tol'],
        alpha=config['lr']['alpha'],
        max_iter=config['lr']['max_iter'],
        early_stop=config['lr']['early_stop'],
        batch_size=config['lr']['batch_size'],
        learning_rate=config['lr']['learning_rate'],
        decay=config['lr']['decay'],
        init_method=config['lr']['init_method'],
        cv_param=config['lr']['cv_param']
    )

    # Step 5: Evaluation
    eval_0 = Evaluation(name="eval_0", eval_type="binary")

    # Build pipeline
    pipeline = Pipeline()
    pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=reader_0.output.data)
    pipeline.add_component(scale_0, data=data_transform_0.output.data)
    pipeline.add_component(lr_0, data=scale_0.output.data)
    pipeline.add_component(eval_0, data=lr_0.output.data)

    # Compile and fit pipeline
    pipeline.compile()
    pipeline.fit()

    # Deploy selected components
    pipeline.deploy([data_transform_0, scale_0, lr_0])

    # Prediction pipeline
    predict_pipe = predict_pipeline.PredictPipeline()
    predict_pipe.add_component(reader_0, data=predict_pipe.data)
    predict_pipe.add_component(data_transform_0, data=reader_0.output.data)
    predict_pipe.add_component(scale_0, data=data_transform_0.output.data)
    predict_pipe.add_component(lr_0, data=scale_0.output.data)

    # Compile prediction pipeline
    predict_pipe.compile()

    # Prediction
    predict_job = predict_pipe.fit()

    # Save DSL and config
    with open('predict_pipeline_dsl.json', 'w') as f:
        json.dump(predict_pipe.get_dsl(), f, indent=4)
    with open('predict_pipeline_conf.json', 'w') as f:
        json.dump(predict_pipe.get_conf(), f, indent=4)

    # Print summaries
    print("Logistic Regression Summary:")
    print(json.dumps(pipeline.get_component("lr_0").get_summary(), indent=4))
    print("Evaluation Summary:")
    print(json.dumps(pipeline.get_component("eval_0").get_summary(), indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config)
```