```python
import argparse
import qlib
from qlib.workflow import R
from qlib.workflow.online.manager import OnlineManager
from qlib.workflow.task.collect import collector_class_name
from qlib.workflow.task.normalize import normalize_class_name
from qlib.workflow.task.utils import _train_rolling_tasks
from qlib.utils.class_casting import class_casting
from qlib.utils import reset_parameters
from qlib.workflow.task.utils import gen_following_tasks
from qlib.workflow.task.utils import get_instance_with_graph_parameters
from qlib.workflow.task.utils import _get_file_name_with_version
from qlib.workflow.task.utils import storage_name, file_name
from qlib.workflow.task.utils import train_and_test
from qlib.workflow.task.utils import end_task_train
from qlib.workflow.task.utils import run_task, task_train
from qlib.workflow.task.utils import prepare_tasks, first_tasks
from qlib.workflow.task.utils import And, __and__
from qlib.workflow.task.utils import setUpClass, tearDownClass

class RollingOnlineManager:
    def __init__(
        self,
        provider_uri,
        region,
        experiment_name,
        task_url,
        task_db_name,
        task_pool_name,
        rolling_step,
        start_time,
        end_time,
        tasks,
        trainer,
    ):
        self.provider_uri = provider_uri
        self.region = region
        self.experiment_name = experiment_name
        self.task_url = task_url
        self.task_db_name = task_db_name
        self.task_pool_name = task_pool_name
        self.rolling_step = rolling_step
        self.start_time = start_time
        self.end_time = end_time
        self.tasks = tasks
        self.trainer = trainer
        self.exp = None
        self.rolling_tasks = None
        self.results = None
        self.signals = None
        self.backtest_res = None
        self.risk_res = None

        qlib.init(provider_uri=self.provider_uri, region=self.region)

    def reset_experiment(self):
        R.set_uri(self.task_url)
        self.exp = R.get_exp(experiment_name=self.experiment_name)
        self.exp.reset()
        reset_parameters()

    def run_workflow(self):
        self.reset_experiment()
        self.simulate()
        self.collect_results()
        self.get_signals()
        self.perform_backtesting()
        self.risk_analysis()

    def simulate(self):
        self.rolling_tasks = _train_rolling_tasks(
            tasks=self.tasks,
            trainer=self.trainer,
            rolling_step=self.rolling_step,
            start_time=self.start_time,
            end_time=self.end_time,
            experiment_name=self.experiment_name,
            task_pool_name=self.task_pool_name,
            task_db_name=self.task_db_name,
        )

    def collect_results(self):
        collector = class_casting(collector_class_name, {})
        self.results = collector.collect(self.rolling_tasks)

    def get_signals(self):
        normalize = class_casting(normalize_class_name, {})
        self.signals = normalize.normalize(self.results)

    def perform_backtesting(self):
        self.backtest_res = train_and_test(self.signals)

    def risk_analysis(self):
        # Placeholder for risk analysis logic
        self.risk_res = {"risk": "analysis_result"}

    def train_tasks_external(self, task_list):
        for task in task_list:
            run_task(task, self.trainer)

def main():
    parser = argparse.ArgumentParser(description="Rolling OnlineManager Simulator")
    parser.add_argument("--provider_uri", type=str, required=True)
    parser.add_argument("--region", type=str, default="cn")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--task_url", type=str, required=True)
    parser.add_argument("--task_db_name", type=str, required=True)
    parser.add_argument("--task_pool_name", type=str, required=True)
    parser.add_argument("--rolling_step", type=int, default=20)
    parser.add_argument("--start_time", type=str, required=True)
    parser.add_argument("--end_time", type=str, required=True)
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--trainer", type=str, required=True)
    args = parser.parse_args()

    # Assume tasks and trainer are passed as importable python dicts/objects
    import importlib.util
    import ast

    def load_obj_from_str(obj_str):
        try:
            return ast.literal_eval(obj_str)
        except Exception:
            spec = importlib.util.spec_from_file_location("module.name", obj_str)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            return foo.obj

    tasks = load_obj_from_str(args.tasks)
    trainer = load_obj_from_str(args.trainer)

    manager = RollingOnlineManager(
        provider_uri=args.provider_uri,
        region=args.region,
        experiment_name=args.experiment_name,
        task_url=args.task_url,
        task_db_name=args.task_db_name,
        task_pool_name=args.task_pool_name,
        rolling_step=args.rolling_step,
        start_time=args.start_time,
        end_time=args.end_time,
        tasks=tasks,
        trainer=trainer,
    )
    manager.run_workflow()
    print("Backtest Results:", manager.backtest_res)
    print("Risk Analysis:", manager.risk_res)

if __name__ == "__main__":
    main()
```