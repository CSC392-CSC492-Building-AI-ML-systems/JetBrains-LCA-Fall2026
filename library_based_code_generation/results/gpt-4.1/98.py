```python
import argparse
from hydro.scenario_creator import scenario_creator, scenario_denouement
from hydro.utils import create_scenario_names, create_node_names
from mpisppy.utils import vanilla
from mpisppy.utils.config import Config
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.spoke_xhatlooper import XhatLooper
from mpisppy.spoke_lagrangian import LagrangianOuterBound

def parse_args():
    parser = argparse.ArgumentParser(description="Hydro-MPISppy Runner")
    parser.add_argument("--branching-factors", type=int, nargs='+', required=True, help="Branching factors for the scenario tree")
    parser.add_argument("--num-scens", type=int, required=True, help="Number of scenarios")
    parser.add_argument("--solver-name", type=str, default="gurobi", help="Solver name")
    parser.add_argument("--max-iterations", type=int, default=100, help="Max iterations for the hub")
    parser.add_argument("--lagrangian", action='store_true', help="Use Lagrangian bound spoke")
    parser.add_argument("--xhatlooper", action='store_true', help="Use XhatLooper bound spoke")
    parser.add_argument("--write-solutions", action='store_true', help="Write solutions to files")
    parser.add_argument("--output-prefix", type=str, default="hydro_output", help="Prefix for output files")
    args = parser.parse_args()
    cfg = Config()
    cfg.branching_factors = args.branching_factors
    cfg.num_scens = args.num_scens
    cfg.solver_name = args.solver_name
    cfg.max_iterations = args.max_iterations
    cfg.lagrangian = args.lagrangian
    cfg.xhatlooper = args.xhatlooper
    cfg.write_solutions = args.write_solutions
    cfg.output_prefix = args.output_prefix
    return cfg

def main():
    cfg = parse_args()
    if len(cfg.branching_factors) < 1:
        raise ValueError("At least one branching factor is required.")
    node_names = create_node_names(cfg.branching_factors)
    scenario_names = create_scenario_names(cfg.num_scens)
    scenario_creator_kwargs = {
        "branching_factors": cfg.branching_factors,
        "solver_name": cfg.solver_name,
    }
    hub_dict = vanilla.ph_hub(
        scenario_names=scenario_names,
        scenario_creator=scenario_creator,
        scenario_denouement=scenario_denouement,
        all_nodenames=node_names,
        scenario_creator_kwargs=scenario_creator_kwargs,
        ph_options={
            "solvername": cfg.solver_name,
            "PHIterLimit": cfg.max_iterations,
        },
    )
    spokes = []
    if cfg.lagrangian:
        lagrangian_spoke = LagrangianOuterBound.make_spoke(
            scenario_names=scenario_names,
            scenario_creator=scenario_creator,
            scenario_denouement=scenario_denouement,
            all_nodenames=node_names,
            scenario_creator_kwargs=scenario_creator_kwargs,
            lagrangian_options={
                "solvername": cfg.solver_name,
            },
        )
        spokes.append(lagrangian_spoke)
    if cfg.xhatlooper:
        xhatlooper_spoke = XhatLooper.make_spoke(
            scenario_names=scenario_names,
            scenario_creator=scenario_creator,
            scenario_denouement=scenario_denouement,
            all_nodenames=node_names,
            scenario_creator_kwargs=scenario_creator_kwargs,
            xhat_looper_options={
                "solvername": cfg.solver_name,
            },
        )
        spokes.append(xhatlooper_spoke)
    wheel = WheelSpinner(hub_dict, list_of_spoke_dict=spokes)
    wheel.spin()
    print(f"Best Inner Bound: {wheel.BestInnerBound}")
    print(f"Best Outer Bound: {wheel.BestOuterBound}")
    if cfg.write_solutions:
        vanilla.write_first_stage_solution(wheel, cfg.output_prefix + "_first_stage.csv")
        vanilla.write_full_tree_solution(wheel, cfg.output_prefix + "_full_tree.csv")

if __name__ == "__main__":
    main()
```