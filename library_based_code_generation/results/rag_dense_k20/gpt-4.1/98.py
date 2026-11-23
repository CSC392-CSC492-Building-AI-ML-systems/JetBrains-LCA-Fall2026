```python
import sys
from hydro import (
    generate_scenario_tree,
    construct_instances_for_scenario_tree,
    pysp_scenario_tree_model_callback,
    scenario_creator_wrapper,
    build_model_for_scenario,
    write_spin_the_wheel_tree_solution,
    MakeAllScenarioTreeNodes,
    scenario_creator_callback,
    CreateAbstractScenarioTreeModel,
    scenario_tree_solution_writer,
    scenario_creator,
    write_spin_the_wheel_first_stage_solution,
    CreateConcreteTwoStageScenarioTreeModel,
    ScenarioCycler,
    scenario_names_creator,
    TestScenarioTree,
    branching_factors_from_numscens,
)
from mpisppy.utils import vanilla
from mpisppy.utils import config
from mpisppy.utils import baseparsers
from mpisppy.utils import spin_the_wheel
from mpisppy.utils import lagrangian
from mpisppy.utils import xhatlooper

def parse_args_and_create_config():
    parser = baseparsers.make_parser()
    parser.add_argument("--branching-factors", nargs='+', type=int, required=True,
                        help="List of branching factors for each stage")
    parser.add_argument("--num-scens", type=int, required=True,
                        help="Total number of scenarios")
    parser.add_argument("--lagrangian", action='store_true',
                        help="Include Lagrangian bound spoke")
    parser.add_argument("--xhatlooper", action='store_true',
                        help="Include xhat looper bound spoke")
    parser.add_argument("--write-solutions", action='store_true',
                        help="Write first stage and full tree solutions")
    config.add_common_config_arguments(parser)
    args = parser.parse_args()
    cfg = config.Config()
    cfg.branching_factors = args.branching_factors
    cfg.num_scens = args.num_scens
    cfg.lagrangian = args.lagrangian
    cfg.xhatlooper = args.xhatlooper
    cfg.write_solutions = args.write_solutions
    config.fill_common_config_from_args(cfg, args)
    return cfg

def main():
    cfg = parse_args_and_create_config()
    branching_factors = cfg.branching_factors
    num_stages = len(branching_factors) + 1
    if len(branching_factors_from_numscens(cfg.num_scens, num_stages)) != len(branching_factors):
        print("Error: Length of branching factors does not match the number of stages minus one.")
        sys.exit(1)
    node_names = MakeAllScenarioTreeNodes(branching_factors)
    scenario_names = scenario_names_creator(branching_factors)
    scenario_creator_func = scenario_creator_wrapper(scenario_creator)
    scenario_denouement_func = scenario_tree_solution_writer

    hub_dict = vanilla.hub_dict(
        cfg,
        scenario_names=scenario_names,
        scenario_creator=scenario_creator_func,
        scenario_denouement=scenario_denouement_func,
        all_nodenames=node_names,
    )

    spokes = []
    if cfg.lagrangian:
        lagrangian_spoke = lagrangian.spoke_dict(
            cfg,
            scenario_names=scenario_names,
            scenario_creator=scenario_creator_func,
            scenario_denouement=scenario_denouement_func,
            all_nodenames=node_names,
        )
        spokes.append(lagrangian_spoke)
    if cfg.xhatlooper:
        xhatlooper_spoke = xhatlooper.spoke_dict(
            cfg,
            scenario_names=scenario_names,
            scenario_creator=scenario_creator_func,
            scenario_denouement=scenario_denouement_func,
            all_nodenames=node_names,
        )
        spokes.append(xhatlooper_spoke)

    wheel = spin_the_wheel.SpinTheWheel(hub_dict, spokes)
    wheel.spin()
    print(f"Best Inner Bound: {wheel.BestInnerBound}")
    print(f"Best Outer Bound: {wheel.BestOuterBound}")

    if cfg.write_solutions:
        write_spin_the_wheel_first_stage_solution(wheel)
        write_spin_the_wheel_tree_solution(wheel)

if __name__ == "__main__":
    main()
```