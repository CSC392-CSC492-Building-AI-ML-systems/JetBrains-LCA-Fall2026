```python
import argparse
import qlib
from qlib.workflow import R
from qlib.workflow.online.manager import OnlineManager
from qlib.workflow.task.collect import Collector
from qlib.workflow.task.utils import TaskManager
from qlib.workflow.task.gen import TaskGen
from qlib.workflow.task.utils import create_task
from qlib.utils.time import Freq
from qlib.utils import init_instance_by_config
import os
import datetime

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
        self.qlib_inited = False
        self.om = None
        self.task_manager = None

    def _init_qlib(self):
        if not self.qlib_inited:
            qlib.init(provider_uri=self.provider_uri, region=self.region)
            self.qlib_inited = True

    def reset_experiment(self):
        self._init_qlib()
        if os.path.exists(self.task_db_name):
            os.remove(self.task_db_name)
        self.om = OnlineManager(
            task_url=self.task_url,
            task_db_name=self.task_db_name,
            task_pool_name=self.task_pool_name,
            rolling_step=self.rolling_step,
            start_time=self.start_time,
            end_time=self.end_time,
            tasks=self.tasks,
            trainer=self.trainer,
        )
        self.task_manager = TaskManager(self.task_db_name, self.task_pool_name)
        print("Experiment reset.")

    def run_workflow(self):
        self._init_qlib()
        self.reset_experiment()
        print("Preparing tasks...")
        self.om.prepare_tasks()
        print("Training rolling tasks...")
        self.om._train_rolling_tasks()
        print("Collecting results...")
        collector = Collector(self.task_db_name, self.task_pool_name)
        collector.collect()
        print("Getting signals...")
        signals = self.om.get_signals()
        print("Performing backtesting...")
        report_normal, positions_normal = self.om.backtest(signals)
        print("Performing risk analysis...")
        analysis = self.om.risk_analysis(report_normal, positions_normal)
        print("Workflow finished.")
        return {
            "signals": signals,
            "report": report_normal,
            "positions": positions_normal,
            "analysis": analysis,
        }

    def train_tasks_external(self):
        self._init_qlib()
        self.task_manager = TaskManager(self.task_db_name, self.task_pool_name)
        tasks = self.task_manager.get_unfinished_tasks()
        for task in tasks:
            print(f"Training task {task['task_id']} ...")
            self.om.task_train(task)
            print(f"Task {task['task_id']} finished.")

    @staticmethod
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
        parser.add_argument("--mode", type=str, choices=["auto", "external"], default="auto")
        args = parser.parse_args()

        # Convert tasks and trainer from string to dict if needed
        import json
        try:
            tasks = json.loads(args.tasks)
        except Exception:
            tasks = args.tasks
        try:
            trainer = json.loads(args.trainer)
        except Exception:
            trainer = args.trainer

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

        if args.mode == "auto":
            results = manager.run_workflow()
            print("Signals:", results["signals"])
            print("Report:", results["report"])
            print("Positions:", results["positions"])
            print("Analysis:", results["analysis"])
        elif args.mode == "external":
            manager.train_tasks_external()

if __name__ == "__main__":
    RollingOnlineManager.main()
```