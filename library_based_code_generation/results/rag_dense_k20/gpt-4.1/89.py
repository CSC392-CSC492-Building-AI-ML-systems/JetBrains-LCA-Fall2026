```python
import argparse
import json
from fate.ml.pipeline import PipelineJob
from fate.ml.components import (
    Reader,
    DataTransform,
    FeatureScale,
    LogisticRegression,
    Evaluation
)
from fate.ml.runtime import predict_pipeline

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
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(
        table=guest_table
    )
    reader_0.get_party_instance(role='host', party_id=host).component_param(
        table=host_table
    )

    # Step 2: Data Transform
    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role='guest', party_id=guest).component_param(
        with_label=True,
        output_format="dense"
    )
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(
        with_label=False
    )

    # Step 3: Feature Scale
    scale_0 = FeatureScale(name="scale_0")
    scale_0.get_party_instance(role='guest', party_id=guest).component_param(
        method="standard_scale"
    )
    scale_0.get_party_instance(role='host', party_id=host).component_param(
        method="standard_scale"
    )

    # Step 4: Logistic Regression
    lr_0 = LogisticRegression(name="lr_0")
    lr_0.get_party_instance(role='guest', party_id=guest).component_param(
        penalty="L2",
        optimizer="sgd",
        tol=1e-4,
        alpha=0.01,
        max_iter=30,
        early_stop="diff",
        batch_size=320,
        learning_rate=0.15,
        decay=0.0,
        init_method="random_uniform",
        cv_param={
            "n_splits": 5,
            "shuffle": True,
            "random_seed": 42,
            "need_cv": True
        }
    )
    lr_0.get_party_instance(role='host', party_id=host).component_param()

    # Step 5: Evaluation
    eval_0 = Evaluation(name="eval_0")
    eval_0.get_party_instance(role='guest', party_id=guest).component_param(
        eval_type="binary"
    )

    # Build pipeline
    pipeline = PipelineJob()
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=reader_0.output.data)
    pipeline.add_component(scale_0, data=data_transform_0.output.data)
    pipeline.add_component(lr_0, data=scale_0.output.data)
    pipeline.add_component(eval_0, data=lr_0.output.data)

    pipeline.compile()
    pipeline.fit()

    # Deploy selected components
    pipeline.deploy([data_transform_0, scale_0, lr_0])

    # Prediction pipeline
    predict_reader = Reader(name="predict_reader")
    predict_reader.get_party_instance(role='guest', party_id=guest).component_param(
        table=config['data']['guest_predict']
    )
    predict_reader.get_party_instance(role='host', party_id=host).component_param(
        table=config['data']['host_predict']
    )

    predict_pipeline_inst = predict_pipeline.PredictPipeline()
    predict_pipeline_inst.add_component(predict_reader)
    predict_pipeline_inst.add_component(
        pipeline.get_deploy_component(data_transform_0),
        data=predict_reader.output.data
    )
    predict_pipeline_inst.add_component(
        pipeline.get_deploy_component(scale_0),
        data=predict_pipeline_inst.get_component(data_transform_0.name).output.data
    )
    predict_pipeline_inst.add_component(
        pipeline.get_deploy_component(lr_0),
        data=predict_pipeline_inst.get_component(scale_0.name).output.data
    )

    predict_pipeline_inst.compile()
    predict_pipeline_inst.predict()

    # Save DSL and config
    with open("predict_pipeline_dsl.json", "w") as f:
        json.dump(predict_pipeline_inst.get_dsl(), f, indent=4)
    with open("predict_pipeline_conf.json", "w") as f:
        json.dump(predict_pipeline_inst.get_conf(), f, indent=4)

    # Print summaries
    print("Logistic Regression Summary:")
    print(json.dumps(lr_0.get_summary(), indent=4))
    print("Evaluation Summary:")
    print(json.dumps(eval_0.get_summary(), indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)
```