```python
import argparse
from hydro import scenario_creator, scenario_denouement, production_cost_function
from mpisppy.utils import vanilla
from mpisppy.utils import spin_the_wheel
from mpisppy.utils import write_spin_the_wheel_first_stage_solution, write_spin_the_wheel_tree_solution
from mpisppy.utils import baseparsers
from mpisppy.utils import config
from mpisppy.utils import lagrangian
from mpisppy.utils import xhat_looper
from mpisppy.utils import wheelspinner

def parse_args_and_create_config():
    parser = argparse.ArgumentParser(description="Hydro-MPISppy Example")
    baseparsers.add_common_args(parser)
    parser.add_argument("--branching-factors", type=int, nargs='+', required=True, help="Branching factors for the scenario tree")
    parser.add_argument("--num-scens", type=int, default=3, help="Number of scenarios")
    parser.add_argument("--lagrangian", action="store_true", help="Use Lagrangian bound spoke")
    parser.add_argument("--xhatlooper", action="store_true", help="Use xhat looper bound spoke")
    parser.add_argument("--write-solutions", action="store_true", help="Write first stage and tree solutions")
    config.add_to_parser(parser)
    args = parser.parse_args()
    cfg = config.Config()
    cfg.populate_from_args(args)
    return args, cfg

def main():
    args, cfg = parse_args_and_create_config()
    branching_factors = args.branching_factors
    if len(branching_factors) < 1:
        raise ValueError("At least one branching factor must be specified.")
    node_names = ["ROOT"]
    for i, bf in enumerate(branching_factors):
        node_names.extend([f"NODE_{i}_{j}" for j in range(bf)])
    scenario_names = [f"scen{i+1}" for i in range(args.num_scens)]
    scenario_creator_kwargs = {"production_cost_function": production_cost_function}
    scenario_creator_func = scenario_creator
    scenario_denouement_func = scenario_denouement

    hub_dict = vanilla.ph_hub_dict(
        cfg,
        scenario_creator_func,
        scenario_denouement_func,
        scenario_names,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    spokes = []
    if args.lagrangian:
        lagrangian_spoke = lagrangian.lagrangian_bound_spoke_dict(
            cfg,
            scenario_creator_func,
            scenario_denouement_func,
            scenario_names,
            scenario_creator_kwargs=scenario_creator_kwargs,
        )
        spokes.append(lagrangian_spoke)
    if args.xhatlooper:
        xhatlooper_spoke = xhat_looper.xhatlooper_spoke_dict(
            cfg,
            scenario_creator_func,
            scenario_denouement_func,
            scenario_names,
            scenario_creator_kwargs=scenario_creator_kwargs,
        )
        spokes.append(xhatlooper_spoke)

    wheel = wheelspinner.WheelSpinner(hub_dict, spokes)
    wheel.spin_the_wheel()
    print(f"Best Inner Bound: {wheel.BestInnerBound}")
    print(f"Best Outer Bound: {wheel.BestOuterBound}")

    if args.write_solutions:
        write_spin_the_wheel_first_stage_solution(wheel)
        write_spin_the_wheel_tree_solution(wheel)

if __name__ == "__main__":
    main()
```