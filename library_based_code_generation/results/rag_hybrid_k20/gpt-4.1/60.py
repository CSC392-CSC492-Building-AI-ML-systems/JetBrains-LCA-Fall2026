```python
import basilisk
from basilisk.simulation import test_MonteCarloSimulation
from basilisk.simulation import runInitialConditions
from basilisk.simulation import addRetentionPolicy
from basilisk.simulation import setShouldDisperseSeeds
from basilisk.simulation import setShouldArchiveParameters
from basilisk.simulation import reRunCases

def rerun_monte_carlo_simulations(
    scenario_name,
    num_processes=1,
    run_numbers=None,
    retention_policies=None,
    ic_dir="initialConditions",
    archive_dir="archive",
    execution_count=1,
    disperse_seeds=True,
    archive_parameters=True
):
    # Set up Monte Carlo controller
    mc = test_MonteCarloSimulation()
    mc.setScenarioName(scenario_name)
    mc.setNumProcesses(num_processes)
    mc.setInitialConditionsDir(ic_dir)
    mc.setArchiveDir(archive_dir)
    mc.setExecutionCount(execution_count)
    setShouldDisperseSeeds(mc, disperse_seeds)
    setShouldArchiveParameters(mc, archive_parameters)

    # Add retention policies if provided
    if retention_policies:
        for policy in retention_policies:
            addRetentionPolicy(mc, policy)

    # If run_numbers is specified, extract subset and rerun
    if run_numbers is not None:
        reRunCases(mc, run_numbers)
    else:
        runInitialConditions(mc)

if __name__ == "__main__":
    rerun_monte_carlo_simulations(
        scenario_name="defaultScenario",
        num_processes=4,
        run_numbers=[1, 2, 3],
        retention_policies=[],
        ic_dir="initialConditions",
        archive_dir="archive",
        execution_count=1,
        disperse_seeds=True,
        archive_parameters=True
    )
```