```python
import argparse
import os
from mpi4py import MPI
import mpi_sppy.utils.sputils as sputils
from mpi_sppy.spin_the_wheel import WheelSpinner, spin_the_wheel
from mpi_sppy.utils import write_spin_the_wheel_tree_solution, write_spin_the_wheel_first_stage_solution
from mpi_sppy.phbase import PHBase
from mpi_sppy.extensions.fwph import FWPH
from mpi_sppy.extensions.xhatlooper import XhatLooper
from mpi_sppy.extensions.xhatshuffler import XhatShuffler
from mpi_sppy.extensions.xhatclosest import XhatClosest
from mpi_sppy.extensions.cross_scen_extension import CrossScenarioExtension
from mpi_sppy.utils import baseparsers

def parse_args():
    parser = argparse.ArgumentParser(description="Run mpi-sppy scenario optimization.")
    parser.add_argument("--num-scenarios", type=int, required=True, help="Number of scenarios to use.")
    parser.add_argument("--solution-dir", type=str, default=None, help="Directory to write solution files.")
    parser.add_argument("--extension", type=str, choices=["fwph", "xhatlooper", "xhatshuffler", "xhatclosest", "crossscenario"], default=None, help="Extension to use.")
    parser.add_argument("--spokes", type=str, nargs='*', default=[], help="List of spokes to use.")
    parser.add_argument("--config-file", type=str, default=None, help="Optional configuration file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--available-scenarios", type=int, nargs='+', default=[2, 4, 8, 16, 32], help="Available scenario counts.")
    return parser.parse_args()

class Config:
    def __init__(self, args):
        self.num_scenarios = args.num_scenarios
        self.solution_dir = args.solution_dir
        self.extension = args.extension
        self.spokes = args.spokes
        self.config_file = args.config_file
        self.seed = args.seed
        self.available_scenarios = args.available_scenarios

def scenario_creator(scenario_name, **kwargs):
    # Placeholder for actual scenario creation logic
    return {}

def main():
    args = parse_args()
    config = Config(args)

    if config.num_scenarios not in config.available_scenarios:
        raise ValueError(f"Number of scenarios {config.num_scenarios} not in available scenarios {config.available_scenarios}")

    scenario_names = [f"Scenario{i+1}" for i in range(config.num_scenarios)]

    extensions = None
    if config.extension == "fwph":
        extensions = FWPH
    elif config.extension == "xhatlooper":
        extensions = XhatLooper
    elif config.extension == "xhatshuffler":
        extensions = XhatShuffler
    elif config.extension == "xhatclosest":
        extensions = XhatClosest
    elif config.extension == "crossscenario":
        extensions = CrossScenarioExtension

    spokes = []
    for spoke in config.spokes:
        if spoke == "fwph":
            spokes.append(FWPH)
        elif spoke == "xhatlooper":
            spokes.append(XhatLooper)
        elif spoke == "xhatshuffler":
            spokes.append(XhatShuffler)
        elif spoke == "xhatclosest":
            spokes.append(XhatClosest)
        elif spoke == "crossscenario":
            spokes.append(CrossScenarioExtension)

    hub_dict = {
        "scenario_creator": scenario_creator,
        "scenario_names": scenario_names,
        "extensions": extensions,
        "all_n": config.num_scenarios,
        "seed": config.seed,
    }

    spinner = WheelSpinner(hub_dict=hub_dict, list_of_spoke_dict=[{"spoke": s} for s in spokes])
    spin_the_wheel(spinner)

    if config.solution_dir is not None:
        os.makedirs(config.solution_dir, exist_ok=True)
        write_spin_the_wheel_first_stage_solution(spinner, config.solution_dir)
        write_spin_the_wheel_tree_solution(spinner, config.solution_dir)

if __name__ == "__main__":
    main()
```