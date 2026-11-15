```python
import argparse
import qlib
from qlib.workflow import R
from qlib.workflow.online_manager import OnlineManager
from qlib.workflow.task.collect import Collector
from qlib.workflow.task.utils import get_task_url
from qlib.contrib.evaluate import risk_analysis, backtest
import os
import pandas as pd

class RollingOnlineManagerSim:
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

    def reset_experiment(self):
        if os.path.exists(self.task_db_name):
            os.remove(self.task_db_name)
        R.set_uri(self.task_url)
        R.set_experiment(self.experiment_name)
        self.om = OnlineManager(
            rolling_step=self.rolling_step,
            start_time=self.start_time,
            end_time=self.end_time,
            task_pool_name=self.task_pool_name,
            task_db_name=self.task_db_name,
            tasks=self.tasks,
            trainer=self.trainer,
        )

    def run_workflow(self):
        self.reset_experiment()
        self.om.simulate()
        results = self.collect_results()
        signals = self.get_signals(results)
        bt_res = self.perform_backtest(signals)
        risk = self.perform_risk_analysis(bt_res)
        return {
            "results": results,
            "signals": signals,
            "backtest": bt_res,
            "risk": risk,
        }

    def train_tasks_external(self):
        collector = Collector(
            task_pool_name=self.task_pool_name,
            task_db_name=self.task_db_name,
        )
        collector.collect()

    def collect_results(self):
        records = []
        for record in R.list_recorders(experiment_name=self.experiment_name):
            rid = record["id"]
            recorder = R.get_recorder(rid, experiment_name=self.experiment_name)
            pred_path = recorder.list_outputs().get("pred.pkl")
            if pred_path:
                pred = recorder.load_object("pred.pkl")
                records.append(pred)
        return records

    def get_signals(self, results):
        signals = pd.concat(results)
        signals = signals.sort_index()
        return signals

    def perform_backtest(self, signals):
        # Example config, should be replaced with actual strategy config
        strategy_config = {
            "topk": 50,
            "n_drop": 5,
        }
        bt_res = backtest(
            pred=signals,
            strategy_config=strategy_config,
            verbose=False,
        )
        return bt_res

    def perform_risk_analysis(self, bt_res):
        analysis = risk_analysis(
            pd.Series(bt_res["return"]),
            freq="day"
        )
        return analysis

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

    qlib.init(provider_uri=args.provider_uri, region=args.region)

    import importlib
    tasks = eval(args.tasks)
    trainer = eval(args.trainer)

    manager = RollingOnlineManagerSim(
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

    results = manager.run_workflow()
    print("Risk Analysis:")
    print(results["risk"])

if __name__ == "__main__":
    main()
```