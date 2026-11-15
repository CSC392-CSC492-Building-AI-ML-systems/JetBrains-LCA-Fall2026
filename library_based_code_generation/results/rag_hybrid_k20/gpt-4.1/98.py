```python
import argparse
from hydro import (
    scenario_creator,
    scenario_denouement,
    create_nodenames_from_branching_factors,
    scenario_names_creator,
    write_spin_the_wheel_first_stage_solution,
    write_spin_the_wheel_tree_solution,
)
from mpisppy.utils import vanilla
from mpisppy.utils import baseparsers
from mpisppy.utils import config
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.cylinders import vanilla as vanilla_cylinders
from mpisppy.cylinders import lagrangianbound, xhatlooperbound

def parse_args_and_create_config():
    parser = argparse.ArgumentParser(description="Hydro-MPISppy runner")
    baseparsers.add_common_args(parser)
    parser.add_argument("--branching-factors", type=int, nargs='+', required=True,
                        help="List of branching factors for each stage")
    parser.add_argument("--num-scens", type=int, default=None,
                        help="Number of scenarios (overrides branching factors if given)")
    parser.add_argument("--with-lagrangian", action="store_true",
                        help="Include Lagrangian bound spoke")
    parser.add_argument("--with-xhatlooper", action="store_true",
                        help="Include xhat looper bound spoke")
    parser.add_argument("--write-solutions", action="store_true",
                        help="Write first stage and tree solutions")
    config.add_config_args(parser)
    args = parser.parse_args()
    cfg = config.Config()
    cfg.populate(args)
    return args, cfg

def main():
    args, cfg = parse_args_and_create_config()

    if args.num_scens is not None:
        branching_factors = vanilla.branching_factors_from_numscens(args.num_scens)
    else:
        branching_factors = args.branching_factors

    if len(branching_factors) < 1:
        raise ValueError("At least one branching factor is required.")

    node_names = create_nodenames_from_branching_factors(branching_factors)
    scenario_names = scenario_names_creator(branching_factors)

    scenario_creator_func = scenario_creator
    scenario_denouement_func = scenario_denouement

    hub_dict = vanilla_cylinders.hub_dict.copy()
    hub_dict.update({
        "scen_creator": scenario_creator_func,
        "scen_denouement": scenario_denouement_func,
        "all_scenario_names": scenario_names,
        "cfg": cfg,
    })

    spokes = []

    if args.with_lagrangian:
        lagrangian_spoke = lagrangianbound.spoke_dict.copy()
        lagrangian_spoke.update({
            "scen_creator": scenario_creator_func,
            "scen_denouement": scenario_denouement_func,
            "all_scenario_names": scenario_names,
            "cfg": cfg,
        })
        spokes.append(lagrangian_spoke)

    if args.with_xhatlooper:
        xhatlooper_spoke = xhatlooperbound.spoke_dict.copy()
        xhatlooper_spoke.update({
            "scen_creator": scenario_creator_func,
            "scen_denouement": scenario_denouement_func,
            "all_scenario_names": scenario_names,
            "cfg": cfg,
        })
        spokes.append(xhatlooper_spoke)

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict=spokes)
    wheel.spin()

    print(f"Best Inner Bound: {wheel.BestInnerBound}")
    print(f"Best Outer Bound: {wheel.BestOuterBound}")

    if args.write_solutions:
        write_spin_the_wheel_first_stage_solution(wheel)
        write_spin_the_wheel_tree_solution(wheel)

if __name__ == "__main__":
    main()
```