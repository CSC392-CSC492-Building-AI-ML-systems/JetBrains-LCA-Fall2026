```python
import logging
import numpy as np
import h5py
import yaml

from some_library import (
    BatchRequest,
    ExampleTensorflowTrainSource,
    TestTensorflowTrain,
    TestAddBoundaryDistanceGradients,
    PrintProfilingStats,
    request_batch,
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
    request = BatchRequest()
    request.add(RAW, input_size_world)
    request.add(LABELS, output_size_world)
    request.add(AFFINITIES, output_size_world)
    request.add(LOSS_WEIGHTS, output_size_world)
    request.add(PRED_AFFINITIES, output_size_world)
    request.add(GRADIENTS, output_size_world)

    snapshot_request = BatchRequest()
    snapshot_request.add(RAW, input_size_world)
    snapshot_request.add(LABELS, output_size_world)
    snapshot_request.add(AFFINITIES, output_size_world)
    snapshot_request.add(PRED_AFFINITIES, output_size_world)
    snapshot_request.add(GRADIENTS, output_size_world)

    # Assemble training pipeline
    source = ExampleTensorflowTrainSource(
        hdf5_file='train_data.hdf5',
        raw_key=RAW,
        label_key=LABELS,
        batch_request=request
    )
    pipeline = (
        source +
        # Normalize raw data
        lambda batch: batch.normalize(RAW) +
        # Random location
        lambda batch: batch.random_location() +
        # Augmentations
        lambda batch: batch.augment() +
        # Grow boundary between labels
        TestAddBoundaryDistanceGradients(LABELS, output_size_world) +
        # Convert labels into affinities
        lambda batch: batch.labels_to_affinities(LABELS, AFFINITIES) +
        # Balance labels
        lambda batch: batch.balance_labels(LABELS, LOSS_WEIGHTS) +
        # Pre-cache batches
        lambda batch: batch.precache() +
        # Training iteration
        TestTensorflowTrain(
            raw=RAW,
            labels=AFFINITIES,
            loss_weights=LOSS_WEIGHTS,
            pred_affinities=PRED_AFFINITIES,
            gradients=GRADIENTS
        ) +
        # Save batch for inspection
        lambda batch: batch.save_hdf5('snapshot.hdf5', snapshot_request) +
        # Print profiling stats
        PrintProfilingStats(every=10)
    )

    logger.info("Starting training for %d iterations...", num_iterations)
    for iteration in range(num_iterations):
        batch = request_batch(pipeline, request)
    logger.info("Training finished after %d iterations.", num_iterations)

if __name__ == '__main__':
    train_model(1000)
```