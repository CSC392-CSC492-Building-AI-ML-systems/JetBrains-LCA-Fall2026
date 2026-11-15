```python
import argparse
from hydro import (
    create_nodenames_from_branching_factors,
    scenario_names_creator,
    scenario_creator,
    scenario_creator_wrapper,
    write_spin_the_wheel_first_stage_solution,
    write_spin_the_wheel_tree_solution,
)
from mpisppy.utils import vanilla
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.utils import lagrangian, xhatlooper

def parse_args():
    parser = argparse.ArgumentParser(description="Hydro-MPISppy runner")
    parser.add_argument("--branching-factors", type=int, nargs='+', required=True,
                        help="Branching factors for each stage")
    parser.add_argument("--num-scens", type=int, required=True,
                        help="Number of scenarios")
    parser.add_argument("--lagrangian", action='store_true',
                        help="Use Lagrangian bound spoke")
    parser.add_argument("--xhatlooper", action='store_true',
                        help="Use xhat looper bound spoke")
    parser.add_argument("--write-solutions", action='store_true',
                        help="Write first stage and tree solutions")
    parser.add_argument("--solver", type=str, default="gurobi",
                        help="Solver to use")
    parser.add_argument("--default-rho", type=float, default=1.0,
                        help="Default rho for PH")
    parser.add_argument("--max-iterations", type=int, default=100,
                        help="Max PH iterations")
    parser.add_argument("--output-prefix", type=str, default="hydro_output",
                        help="Prefix for output files")
    return parser.parse_args()

def main():
    config = parse_args()
    branching_factors = config.branching_factors
    num_stages = len(branching_factors) + 1
    if len(branching_factors) < 1:
        raise ValueError("At least one branching factor is required.")
    node_names = create_nodenames_from_branching_factors(branching_factors)
    scenario_names = scenario_names_creator(branching_factors)
    scenario_creator_func = scenario_creator
    scenario_denouement_func = None  # Replace with actual denouement if needed

    scenario_creator_kwargs = {
        "branching_factors": branching_factors,
        "num_stages": num_stages,
    }

    hub_dict = vanilla.ph_hub(
        scenario_names=scenario_names,
        scenario_creator=scenario_creator_func,
        scenario_denouement=scenario_denouement_func,
        all_scenario_names=scenario_names,
        scenario_creator_kwargs=scenario_creator_kwargs,
        ph_options={
            "solvername": config.solver,
            "defaultPHrho": config.default_rho,
            "PHIterLimit": config.max_iterations,
        },
    )

    spokes = []
    if config.lagrangian:
        spokes.append(
            lagrangian.lagrangian_spoke(
                scenario_names=scenario_names,
                scenario_creator=scenario_creator_func,
                scenario_denouement=scenario_denouement_func,
                all_scenario_names=scenario_names,
                scenario_creator_kwargs=scenario_creator_kwargs,
                lagrangian_options={
                    "solvername": config.solver,
                },
            )
        )
    if config.xhatlooper:
        spokes.append(
            xhatlooper.xhatlooper_spoke(
                scenario_names=scenario_names,
                scenario_creator=scenario_creator_func,
                scenario_denouement=scenario_denouement_func,
                all_scenario_names=scenario_names,
                scenario_creator_kwargs=scenario_creator_kwargs,
                xhatlooper_options={
                    "solvername": config.solver,
                },
            )
        )

    wheel = WheelSpinner(hub_dict, spokes)
    wheel.spin()
    print("Best Inner Bound:", wheel.BestInnerBound)
    print("Best Outer Bound:", wheel.BestOuterBound)

    if config.write_solutions:
        write_spin_the_wheel_first_stage_solution(wheel, config.output_prefix + "_first_stage.csv")
        write_spin_the_wheel_tree_solution(wheel, config.output_prefix + "_tree_solution.csv")

if __name__ == "__main__":
    main()
```