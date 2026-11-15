```python
import argparse
import os
from mpi4py import MPI
from mpi_sppy.utils import scenario_names_creator, write_spin_the_wheel_first_stage_solution, write_spin_the_wheel_tree_solution
from mpi_sppy.utils import _make_scenario_names
from mpi_sppy.spin_the_wheel import WheelSpinner
from mpi_sppy.scenario_tree import generate_scenario_tree
from mpi_sppy.scenario_creator import scenario_creator
from mpi_sppy.extensions.xhatbase import Xhat_Eval, Xhat_Shuffle
from mpi_sppy.extensions.fixer import Fixer
from mpi_sppy.extensions.cross_scen_extension import CrossScenarioExtension

def parse_args():
    parser = argparse.ArgumentParser(description="MPI-SPPy scenario runner")
    parser.add_argument("--num-scenarios", type=int, required=True, help="Number of scenarios")
    parser.add_argument("--available-scenarios", type=int, nargs='+', default=[2, 4, 8, 16], help="Available scenario counts")
    parser.add_argument("--solution-dir", type=str, default=None, help="Directory to write solution")
    parser.add_argument("--use-xhat-eval", action='store_true', help="Use Xhat_Eval extension")
    parser.add_argument("--use-xhat-shuffle", action='store_true', help="Use Xhat_Shuffle extension")
    parser.add_argument("--use-fixer", action='store_true', help="Use Fixer extension")
    parser.add_argument("--use-cross-scenario", action='store_true', help="Use CrossScenarioExtension")
    parser.add_argument("--spoke", type=str, nargs='*', default=[], help="List of spokes to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

class Config:
    def __init__(self, args):
        self.num_scenarios = args.num_scenarios
        self.available_scenarios = args.available_scenarios
        self.solution_dir = args.solution_dir
        self.use_xhat_eval = args.use_xhat_eval
        self.use_xhat_shuffle = args.use_xhat_shuffle
        self.use_fixer = args.use_fixer
        self.use_cross_scenario = args.use_cross_scenario
        self.spoke = args.spoke
        self.seed = args.seed

def main():
    args = parse_args()
    config = Config(args)

    if config.num_scenarios not in config.available_scenarios:
        raise ValueError(f"Number of scenarios {config.num_scenarios} not in available scenarios {config.available_scenarios}")

    scenario_names = _make_scenario_names(config.num_scenarios)
    scenario_creator_func = scenario_creator

    extensions = []
    if config.use_xhat_eval:
        extensions.append(Xhat_Eval)
    if config.use_xhat_shuffle:
        extensions.append(Xhat_Shuffle)
    if config.use_fixer:
        extensions.append(Fixer)
    if config.use_cross_scenario:
        extensions.append(CrossScenarioExtension)

    hub_dict = {
        "scen_creator": scenario_creator_func,
        "scen_names": scenario_names,
        "extensions": extensions,
        "seed": config.seed,
    }

    spokes = []
    for spoke_name in config.spoke:
        if spoke_name.lower() == "xhat_eval":
            spokes.append({"extension": Xhat_Eval})
        elif spoke_name.lower() == "xhat_shuffle":
            spokes.append({"extension": Xhat_Shuffle})
        elif spoke_name.lower() == "fixer":
            spokes.append({"extension": Fixer})
        elif spoke_name.lower() == "cross_scenario":
            spokes.append({"extension": CrossScenarioExtension})

    spinner = WheelSpinner(hub_dict, spokes)
    spinner.spin()

    if config.solution_dir:
        os.makedirs(config.solution_dir, exist_ok=True)
        write_spin_the_wheel_first_stage_solution(spinner, config.solution_dir)
        write_spin_the_wheel_tree_solution(spinner, config.solution_dir)

if __name__ == "__main__":
    main()
```