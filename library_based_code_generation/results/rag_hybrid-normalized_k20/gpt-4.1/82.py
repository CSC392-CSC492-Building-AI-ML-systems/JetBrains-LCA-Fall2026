```python
import torch
from monai.networks.nets import UNet
from torch.optim import SGD
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.signal import Signal
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.model import ModelLearnable, ModelDescriptor
from nvflare.apis.training_command_module import TrainingCommandModule
from nvflare.apis.executor import stop_training
from nvflare.apis.training_result import TrainingResult
from nvflare.apis.training_context import _prepare_training_ctx
from nvflare.apis.training_loop import training_loop
from nvflare.apis.model_learner import ModelLearner
from nvflare.apis.model_learner_executor import ModelLearnerExecutor
from nvflare.apis.training_step import training_step
from nvflare.apis.training_topic import TrainingTopic
from nvflare.apis.supervised_learner import SupervisedLearner
from nvflare.apis.global_model_eval import GlobalModelEval

from supervised_monai_prostate_learner import SupervisedMonaiProstateLearner
from ptditto_helper import SupervisedPTDittoHelper


class SupervisedMonaiProstateDittoLearner(SupervisedMonaiProstateLearner):
    def __init__(
        self,
        train_conf: str,
        global_epochs: int,
        personalized_epochs: int,
        task_name: str,
        **kwargs
    ):
        super().__init__(train_conf=train_conf, **kwargs)
        self.global_epochs = global_epochs
        self.personalized_epochs = personalized_epochs
        self.task_name = task_name
        self.pt_ditto_helper = None

    def configure_training(self):
        # Initialize Ditto helper with UNet and SGD optimizer
        model = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        self.pt_ditto_helper = SupervisedPTDittoHelper(
            model=model,
            optimizer=optimizer,
            train_conf=self.train_conf,
            task_name=self.task_name,
        )

    def train(
        self,
        model_learnable: ModelLearnable,
        abort_signal: Signal = None,
        **kwargs
    ):
        try:
            if abort_signal and abort_signal.triggered:
                stop_training()
                return None

            # Step 1: Get global model weights
            global_weights = model_learnable

            # Step 2: Prepare for fedprox loss (if needed)
            self.pt_ditto_helper.prepare_fedprox(global_weights)

            # Step 3: Load Ditto personalized model info
            self.pt_ditto_helper.load_personalized_model_info()

            # Step 4: Local training of reference (global) model
            ref_model_learnable = self.pt_ditto_helper.train_reference_model(
                epochs=self.global_epochs,
                abort_signal=abort_signal,
            )

            # Step 5: Local training of personalized model
            self.pt_ditto_helper.train_personalized_model(
                epochs=self.personalized_epochs,
                abort_signal=abort_signal,
            )

            # Step 6: Return updated weights of the reference model
            updated_weights = self.pt_ditto_helper.get_reference_model_learnable()
            return updated_weights

        except Exception as e:
            stop_training()
            raise e
```