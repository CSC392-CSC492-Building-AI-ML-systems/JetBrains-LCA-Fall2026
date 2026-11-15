```python
from supervised_monai_prostate_learner import SupervisedMonaiProstateLearner
from supervised_pt_ditto_helper import SupervisedPTDittoHelper
from monai.networks.nets import UNet
from torch.optim import SGD
import torch

class SupervisedMonaiProstateDittoLearner(SupervisedMonaiProstateLearner):
    def __init__(self, training_config_filename, global_epochs_per_round, personalized_epochs_per_round, task_name):
        super().__init__(training_config_filename, task_name)
        self.global_epochs_per_round = global_epochs_per_round
        self.personalized_epochs_per_round = personalized_epochs_per_round
        self.task_name = task_name
        self.ditto_helper = None

    def configure_training(self):
        model = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        self.ditto_helper = SupervisedPTDittoHelper(
            model=model,
            optimizer=optimizer,
            config_filename=self.training_config_filename,
            task_name=self.task_name
        )

    def train(self, train_data_loader, val_data_loader, abort_signal=None):
        try:
            if abort_signal is not None and abort_signal.is_set():
                self.stop_training()
                return None

            # Step 1: Get global model weights
            global_weights = self._get_model_weights()

            # Step 2: Prepare for fedprox loss (if needed)
            self.ditto_helper.prepare_fedprox(global_weights)

            # Step 3: Load Ditto personalized model info
            self.ditto_helper.load_personalized_model_info()

            # Step 4: Local training of the reference (global) model
            ref_model_weights = self.ditto_helper.train_reference_model(
                train_data_loader,
                epochs=self.global_epochs_per_round,
                abort_signal=abort_signal
            )

            if abort_signal is not None and abort_signal.is_set():
                self.stop_training()
                return None

            # Step 5: Local training of the personalized model
            self.ditto_helper.train_personalized_model(
                train_data_loader,
                epochs=self.personalized_epochs_per_round,
                abort_signal=abort_signal
            )

            # Step 6: Return updated weights of the reference model
            return ref_model_weights

        except Exception as e:
            self.stop_training()
            raise e
```