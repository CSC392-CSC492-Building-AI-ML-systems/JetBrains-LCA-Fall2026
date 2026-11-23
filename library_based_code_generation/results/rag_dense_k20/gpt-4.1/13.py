```python
import argparse
import qlib
from qlib.workflow import R
from qlib.workflow.online.manager import OnlineManager
from qlib.workflow.online.simulator import SimulatorExecutor, get_simulator_executor
from qlib.workflow.task.utils import task_generator
from qlib.workflow.online.utils import _train_rolling_tasks, _create_qlib_backtest_adapter
from qlib.utils.time import Freq
from qlib.data import D
import os
import pandas as pd

class RollingOnlineManagerSimulator:
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
        self.om = None
        self.simulator_executor = None
        self.results = None
        self.signals = None
        self.backtest_res = None
        self.risk_analysis_res = None

        qlib.init(provider_uri=self.provider_uri, region=self.region)

    def reset_experiment(self):
        if os.path.exists(self.task_db_name):
            os.remove(self.task_db_name)
        R.set_uri(self.task_url)
        self.om = OnlineManager(
            experiment_name=self.experiment_name,
            task_url=self.task_url,
            task_db_name=self.task_db_name,
            task_pool_name=self.task_pool_name,
            rolling_step=self.rolling_step,
            start_time=self.start_time,
            end_time=self.end_time,
            tasks=self.tasks,
            trainer=self.trainer,
        )
        self.simulator_executor = get_simulator_executor(self.om)

    def run_workflow(self):
        self.reset_experiment()
        self.simulate()
        self.collect_results()
        self.get_signals()
        self.perform_backtesting()
        self.risk_analysis()

    def simulate(self):
        self.simulator_executor.simulate()

    def collect_results(self):
        self.results = self.simulator_executor.get_results()

    def get_signals(self):
        self.signals = self.simulator_executor.get_signals()

    def perform_backtesting(self):
        if self.signals is not None:
            backtest_adapter = _create_qlib_backtest_adapter(self.signals)
            self.backtest_res = backtest_adapter.backtest()

    def risk_analysis(self):
        if self.backtest_res is not None:
            self.risk_analysis_res = self.backtest_res.get_analysis()

    def train_tasks_external(self):
        _train_rolling_tasks(
            self.task_db_name,
            self.task_pool_name,
            self.trainer,
            self.tasks,
            self.rolling_step,
            self.start_time,
            self.end_time,
        )

def main():
    parser = argparse.ArgumentParser(description="Rolling OnlineManager Simulator")
    parser.add_argument("--provider_uri", type=str, required=True, help="Qlib data provider uri")
    parser.add_argument("--region", type=str, default="cn", help="Market region")
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--task_url", type=str, required=True, help="Task url for experiment")
    parser.add_argument("--task_db_name", type=str, required=True, help="Task database name")
    parser.add_argument("--task_pool_name", type=str, required=True, help="Task pool name")
    parser.add_argument("--rolling_step", type=int, default=20, help="Rolling step")
    parser.add_argument("--start_time", type=str, required=True, help="Start time (YYYY-MM-DD)")
    parser.add_argument("--end_time", type=str, required=True, help="End time (YYYY-MM-DD)")
    parser.add_argument("--tasks", type=str, required=True, help="Tasks config file (python dict)")
    parser.add_argument("--trainer", type=str, required=True, help="Trainer config file (python dict)")
    parser.add_argument("--train_external", action="store_true", help="Train tasks externally (multiprocessing)")

    args = parser.parse_args()

    import ast
    with open(args.tasks, "r") as f:
        tasks = ast.literal_eval(f.read())
    with open(args.trainer, "r") as f:
        trainer = ast.literal_eval(f.read())

    simulator = RollingOnlineManagerSimulator(
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

    if args.train_external:
        simulator.train_tasks_external()
    else:
        simulator.run_workflow()
        print("Results:", simulator.results)
        print("Signals:", simulator.signals)
        print("Backtest Results:", simulator.backtest_res)
        print("Risk Analysis:", simulator.risk_analysis_res)

if __name__ == "__main__":
    main()
```