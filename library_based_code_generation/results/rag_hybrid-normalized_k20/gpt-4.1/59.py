```python
import logging
import numpy as np
import h5py
import yaml

from some_library import (
    __produce_train_batch,
    test_shift_and_crop4,
    test_shift_and_crop3,
    __produce_predict_batch,
    test_shift_and_crop2,
    test_up_and_downsample,
    _setup_batch,
    PrintProfilingStats,
    TestHdf5Source,
    test_shift_and_crop1,
    TestHdf5Write,
    request_batch,
    BatchRequest,
    TestTensorflowTrain,
    test_jitter_and_random,
    ExampleTensorflowTrainSource,
    _add_to_batch,
    Batch,
    __read_file,
    TestAddBoundaryDistanceGradients
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(num_iterations):
    # Declare array keys
    RAW = 'raw'
    LABELS = 'labels'
    AFFINITIES = 'affinities'
    LOSS_WEIGHTS = 'loss_weights'
    PRED_AFFINITIES = 'pred_affinities'
    GRADIENTS = 'gradients'

    # Read configuration file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    voxel_size = np.array(config['voxel_size'])
    input_size = np.array(config['input_size'])
    output_size = np.array(config['output_size'])
    input_size_world = input_size * voxel_size
    output_size_world = output_size * voxel_size

    # Formulate batch and snapshot requests
    batch_request = BatchRequest()
    batch_request.add(RAW, input_size)
    batch_request.add(LABELS, output_size)
    batch_request.add(AFFINITIES, output_size)
    batch_request.add(LOSS_WEIGHTS, output_size)
    batch_request.add(PRED_AFFINITIES, output_size)
    batch_request.add(GRADIENTS, output_size)

    snapshot_request = BatchRequest()
    snapshot_request.add(RAW, input_size)
    snapshot_request.add(LABELS, output_size)
    snapshot_request.add(AFFINITIES, output_size)
    snapshot_request.add(PRED_AFFINITIES, output_size)
    snapshot_request.add(GRADIENTS, output_size)

    # Assemble training pipeline
    source = TestHdf5Source(
        filename=config['train_data'],
        datasets={
            RAW: config['raw_dataset'],
            LABELS: config['labels_dataset']
        }
    )
    source += test_shift_and_crop1()
    source += test_jitter_and_random()
    source += test_shift_and_crop2()
    source += test_shift_and_crop3()
    source += test_shift_and_crop4()
    source += test_up_and_downsample()
    source += TestAddBoundaryDistanceGradients(LABELS, output_size)
    source += ExampleTensorflowTrainSource(
        raw_key=RAW,
        labels_key=LABELS,
        affinities_key=AFFINITIES,
        loss_weights_key=LOSS_WEIGHTS,
        pred_affinities_key=PRED_AFFINITIES,
        gradients_key=GRADIENTS,
        config=config
    )
    source += TestHdf5Write(
        output_filename='snapshot.hdf',
        dataset_names={
            RAW: 'raw',
            LABELS: 'labels',
            AFFINITIES: 'affinities',
            PRED_AFFINITIES: 'pred_affinities',
            GRADIENTS: 'gradients'
        },
        every=1,
        additional_request=snapshot_request
    )
    source += PrintProfilingStats(every=10)

    logger.info("Starting training for %d iterations...", num_iterations)
    pipeline = source
    with pipeline:
        for i in range(num_iterations):
            batch = pipeline.request_batch(batch_request)
    logger.info("Training finished after %d iterations.", num_iterations)

if __name__ == '__main__':
    train_model(100)
```