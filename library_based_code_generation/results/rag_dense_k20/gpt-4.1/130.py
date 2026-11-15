```python
import argparse
import os
from mpi4py import MPI
from mpi_sppy.utils import config
from mpi_sppy.utils import scenario_names_creator, write_spin_the_wheel_first_stage_solution
from mpi_sppy.spin_the_wheel import WheelSpinner
from mpi_sppy.extensions.xhatbase import Xhat_Eval, XhatLooper
from mpi_sppy.extensions.fixer import Fixer
from mpi_sppy.extensions.cross_scen_cut import CrossScenarioCutExtension
from mpi_sppy.utils import _create_scenarios

def parse_args():
    parser = argparse.ArgumentParser(description="MPI-SPPY scenario runner")
    parser.add_argument("--num-scenarios", type=int, required=True, help="Number of scenarios")
    parser.add_argument("--solver", type=str, default="gurobi", help="Solver to use")
    parser.add_argument("--solution-dir", type=str, default=None, help="Directory to write solution")
    parser.add_argument("--xhatlooper", action="store_true", help="Use XhatLooper extension")
    parser.add_argument("--fixer", action="store_true", help="Use Fixer extension")
    parser.add_argument("--cross-scen-cut", action="store_true", help="Use CrossScenarioCut extension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--available-scenarios", type=int, nargs="+", default=[2, 4, 8, 16, 32], help="Available scenario counts")
    return parser.parse_args()

def setup_config(args):
    cfg = config.Config()
    cfg.num_scens = args.num_scenarios
    cfg.solver_name = args.solver
    cfg.seed = args.seed
    cfg.solution_dir = args.solution_dir
    cfg.xhatlooper = args.xhatlooper
    cfg.fixer = args.fixer
    cfg.cross_scen_cut = args.cross_scen_cut
    return cfg

def main():
    args = parse_args()
    cfg = setup_config(args)

    if cfg.num_scens not in args.available_scenarios:
        raise ValueError(f"Number of scenarios {cfg.num_scens} not in available scenarios {args.available_scenarios}")

    scenario_creator = lambda scenario_name, node_names=None: _create_scenarios(scenario_name, seed=cfg.seed)
    scenario_names = scenario_names_creator(cfg.num_scens)

    extensions = {}
    if cfg.xhatlooper:
        extensions["xhatlooper"] = XhatLooper
    if cfg.fixer:
        extensions["fixer"] = Fixer
    if cfg.cross_scen_cut:
        extensions["cross_scen_cut"] = CrossScenarioCutExtension

    spokes = []
    if cfg.xhatlooper:
        spokes.append(("xhatlooper", XhatLooper))
    if cfg.fixer:
        spokes.append(("fixer", Fixer))
    if cfg.cross_scen_cut:
        spokes.append(("cross_scen_cut", CrossScenarioCutExtension))

    wheel = WheelSpinner(
        hub_dict={
            "options": {
                "solver_name": cfg.solver_name,
                "num_scens": cfg.num_scens,
                "seed": cfg.seed,
            },
            "scenario_creator": scenario_creator,
            "scenario_names": scenario_names,
            "extensions": extensions,
        },
        list_of_spoke_dict=[
            {
                "spoke_class": spoke_class,
                "options": {
                    "solver_name": cfg.solver_name,
                    "num_scens": cfg.num_scens,
                    "seed": cfg.seed,
                },
                "scenario_creator": scenario_creator,
                "scenario_names": scenario_names,
            }
            for _, spoke_class in spokes
        ]
    )

    wheel.spin()

    if cfg.solution_dir is not None and MPI.COMM_WORLD.Get_rank() == 0:
        os.makedirs(cfg.solution_dir, exist_ok=True)
        write_spin_the_wheel_first_stage_solution(wheel, cfg.solution_dir)

if __name__ == "__main__":
    main()
```