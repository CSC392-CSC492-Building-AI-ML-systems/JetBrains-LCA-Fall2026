```python
import basilisk
from basilisk.simulation import test_MonteCarloSimulation, runInitialConditions, addRetentionPoliciesToSim
from basilisk.simulation import setShouldDisperseSeeds, setShouldArchiveParameters, addRetentionPolicy
from basilisk.simulation import reRunCases, VariableRetentionParameters

def rerun_monte_carlo_simulations(
    scenario_name,
    num_processes=1,
    run_numbers=None,
    retention_policies=None,
    ic_dir="initial_conditions",
    archive_dir="archive",
    execution_count=100,
    disperse_seeds=True,
    archive_parameters=True
):
    mc = test_MonteCarloSimulation()
    mc.setScenarioName(scenario_name)
    mc.setNumProcesses(num_processes)
    mc.setInitialConditionsDir(ic_dir)
    mc.setArchiveDir(archive_dir)
    mc.setExecutionCount(execution_count)
    mc.setShouldDisperseSeeds(disperse_seeds)
    mc.setShouldArchiveParameters(archive_parameters)
    if retention_policies:
        for policy in retention_policies:
            addRetentionPolicy(mc, policy)
    if run_numbers is not None:
        subset_mc = VariableRetentionParameters()
        subset_mc.extractSubsetOfRuns(run_numbers)
        reRunCases(mc, subset_mc)
    else:
        runInitialConditions(mc)
    return mc

if __name__ == "__main__":
    rerun_monte_carlo_simulations(
        scenario_name="defaultScenario",
        num_processes=4,
        run_numbers=[1, 2, 3],
        retention_policies=[],
        ic_dir="initial_conditions",
        archive_dir="archive",
        execution_count=10,
        disperse_seeds=True,
        archive_parameters=True
    )
```