```python
import os
from Basilisk.utilities import monteCarloController, monteCarloRetentionPolicy

def rerun_monte_carlo(
    scenario_name,
    num_processes=4,
    run_numbers=None,
    retention_policies=None,
    initial_conditions_dir="initialConditions",
    archive_dir="archive",
    execution_count=100,
    disperse_seeds=True,
    archive_parameters=True
):
    # Set up Monte Carlo controller
    mc = monteCarloController.MonteCarloController()
    mc.setScenarioName(scenario_name)
    mc.setExecutionCount(execution_count)
    mc.setInitialConditionsDir(initial_conditions_dir)
    mc.setArchiveDir(archive_dir)
    mc.setNumProcesses(num_processes)
    mc.setDisperseSeeds(disperse_seeds)
    mc.setArchiveParameters(archive_parameters)

    # Add retention policies
    if retention_policies is not None:
        for policy in retention_policies:
            mc.addRetentionPolicy(policy)
    else:
        # Add a default retention policy if none provided
        mc.addRetentionPolicy(monteCarloRetentionPolicy.MonteCarloRetentionPolicy())

    # Specify run numbers to rerun
    if run_numbers is not None:
        mc.setRunNumbers(run_numbers)

    # Run initial conditions
    mc.runInitialConditions()

if __name__ == "__main__":
    rerun_monte_carlo(
        scenario_name="defaultScenario",
        num_processes=4,
        run_numbers=[1, 2, 3],
        retention_policies=[
            monteCarloRetentionPolicy.MonteCarloRetentionPolicy()
        ],
        initial_conditions_dir="initialConditions",
        archive_dir="archive",
        execution_count=100,
        disperse_seeds=True,
        archive_parameters=True
    )
```