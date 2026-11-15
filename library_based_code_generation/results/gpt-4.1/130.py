```python
import argparse
import os
from mpi4py import MPI
from mpi_sppy.utils.config import Config
from mpi_sppy.utils import baseparsers
from mpi_sppy.spin_the_wheel import WheelSpinner
from mpi_sppy.extensions.xhatbase import Xhat_Eval
from mpi_sppy.extensions.fixer import Fixer
from mpi_sppy.extensions.cross_scen_extension import CrossScenarioExtension
from mpi_sppy.utils.sputils import create_n_scenarios
from mpi_sppy.utils import vanilla
from mpi_sppy.utils import vanilla_trees
from mpi_sppy.utils import vanilla_gurobi
from mpi_sppy.utils import vanilla_cylinders

def parse_args():
    parser = argparse.ArgumentParser(description="MPI-SPPy scenario runner")
    baseparsers.add_common_args(parser)
    parser.add_argument("--num-scenarios", type=int, required=True, help="Number of scenarios")
    parser.add_argument("--solution-dir", type=str, default=None, help="Directory to write solution")
    parser.add_argument("--extension", type=str, choices=["xhat", "fixer", "cross"], default=None, help="Extension to use")
    parser.add_argument("--spoke", type=str, choices=["fwph", "xhatshuffle", "xhatlooper", "lagrangian", "cross"], nargs='*', default=[], help="Spokes to use")
    parser.add_argument("--available-scenarios", type=int, nargs='+', default=[2, 4, 8, 16, 32], help="Available scenario counts")
    return parser.parse_args()

def setup_config(args):
    config = Config()
    config.num_scenarios = args.num_scenarios
    config.solution_dir = args.solution_dir
    config.extension = args.extension
    config.spoke = args.spoke
    config.available_scenarios = args.available_scenarios
    return config

def scenario_creator(scenario_name, **kwargs):
    # Placeholder for actual scenario creation logic
    return {}

def main():
    args = parse_args()
    config = setup_config(args)

    if config.num_scenarios not in config.available_scenarios:
        raise ValueError(f"Number of scenarios {config.num_scenarios} not in available scenarios {config.available_scenarios}")

    scenario_names = [f"Scenario{i+1}" for i in range(config.num_scenarios)]

    extensions = None
    if config.extension == "xhat":
        extensions = Xhat_Eval
    elif config.extension == "fixer":
        extensions = Fixer
    elif config.extension == "cross":
        extensions = CrossScenarioExtension

    spokes = []
    for spoke in config.spoke:
        if spoke == "fwph":
            spokes.append(vanilla_cylinders.fwph_spoke)
        elif spoke == "xhatshuffle":
            spokes.append(vanilla_cylinders.xhatshuffle_spoke)
        elif spoke == "xhatlooper":
            spokes.append(vanilla_cylinders.xhatlooper_spoke)
        elif spoke == "lagrangian":
            spokes.append(vanilla_cylinders.lagrangian_spoke)
        elif spoke == "cross":
            spokes.append(vanilla_cylinders.cross_spoke)

    hub_dict = vanilla.hub_dict
    hub_dict["scen_creator"] = scenario_creator
    hub_dict["all_scenario_names"] = scenario_names
    if extensions is not None:
        hub_dict["extensions"] = extensions

    wheel = WheelSpinner(hub_dict, spokes)
    wheel.spin()

    if config.solution_dir is not None and MPI.COMM_WORLD.Get_rank() == 0:
        os.makedirs(config.solution_dir, exist_ok=True)
        solution_path = os.path.join(config.solution_dir, "solution.json")
        with open(solution_path, "w") as f:
            f.write(str(wheel.get_best_solution()))

if __name__ == "__main__":
    main()
```