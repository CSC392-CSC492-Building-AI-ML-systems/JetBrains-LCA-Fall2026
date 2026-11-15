```python
import argparse
import os
from mpi4py import MPI
from mpi_sppy.utils import scenario_names_creator, write_spin_the_wheel_first_stage_solution, write_spin_the_wheel_tree_solution
from mpi_sppy.utils import scenario_creator_wrapper
from mpi_sppy.spin_the_wheel import WheelSpinner
from mpi_sppy.utils import is_scenario

def parse_args():
    parser = argparse.ArgumentParser(description="MPI-SPPy scenario runner")
    parser.add_argument("--num-scenarios", type=int, required=True, help="Number of scenarios")
    parser.add_argument("--solution-dir", type=str, default=None, help="Directory to write solution")
    parser.add_argument("--extension", type=str, choices=["fixer", "xhatshuffle", "xhatlooper", "none"], default="none", help="Extension to use")
    parser.add_argument("--spokes", nargs='*', default=[], help="List of spokes to use")
    parser.add_argument("--available-scenarios", nargs='+', type=int, default=[2, 4, 8, 16, 32], help="Available scenario counts")
    parser.add_argument("--scenario-prefix", type=str, default="Scenario", help="Prefix for scenario names")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

class Config:
    def __init__(self, args):
        self.num_scenarios = args.num_scenarios
        self.solution_dir = args.solution_dir
        self.extension = args.extension
        self.spokes = args.spokes
        self.available_scenarios = args.available_scenarios
        self.scenario_prefix = args.scenario_prefix
        self.seed = args.seed

def scenario_creator(scenario_name, **kwargs):
    # Placeholder for user-defined scenario creator
    pass

def main():
    args = parse_args()
    config = Config(args)

    if config.num_scenarios not in config.available_scenarios:
        raise ValueError(f"Number of scenarios {config.num_scenarios} not in available scenarios {config.available_scenarios}")

    scenario_names = scenario_names_creator(config.num_scenarios, prefix=config.scenario_prefix)
    scenario_creator_func = scenario_creator_wrapper(scenario_creator)

    extensions = []
    if config.extension == "fixer":
        from mpi_sppy.extensions.fixer import Fixer
        extensions.append(Fixer)
    elif config.extension == "xhatshuffle":
        from mpi_sppy.extensions.xhatshuffle import XhatShuffle
        extensions.append(XhatShuffle)
    elif config.extension == "xhatlooper":
        from mpi_sppy.extensions.xhatlooper import XhatLooper
        extensions.append(XhatLooper)

    spokes = []
    for spoke in config.spokes:
        if spoke == "fwph":
            from mpi_sppy.spoke_fwph import FWPHSpoke
            spokes.append(FWPHSpoke)
        elif spoke == "lagrangian":
            from mpi_sppy.spoke_lagrangian import LagrangianSpoke
            spokes.append(LagrangianSpoke)
        elif spoke == "xhatshuffle":
            from mpi_sppy.spoke_xhatshuffle import XhatShuffleInnerBound
            spokes.append(XhatShuffleInnerBound)
        elif spoke == "xhatlooper":
            from mpi_sppy.spoke_xhatlooper import XhatLooperInnerBound
            spokes.append(XhatLooperInnerBound)
        # Add more spokes as needed

    hub_dict = {
        "scen_creator": scenario_creator_func,
        "all_scenario_names": scenario_names,
        "extensions": extensions,
        "seed": config.seed,
    }

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict=[{"spoke_class": s} for s in spokes])
    wheel.spin()

    if config.solution_dir:
        os.makedirs(config.solution_dir, exist_ok=True)
        write_spin_the_wheel_first_stage_solution(wheel, config.solution_dir)
        write_spin_the_wheel_tree_solution(wheel, config.solution_dir)

if __name__ == "__main__":
    main()
```