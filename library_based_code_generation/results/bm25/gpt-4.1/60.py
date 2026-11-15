```python
import os
import sys
from Basilisk.utilities import MonteCarloController
from Basilisk.utilities.MonteCarlo import RetentionPolicy

def rerun_monte_carlo_sims(
    scenario_name,
    num_processes=1,
    run_numbers=None,
    retention_policies=None,
    ic_dir="initialConditions",
    archive_dir="archive",
    execution_count=None,
    disperse_seeds=False,
    archive_parameters=False
):
    # Set up Monte Carlo controller
    mc = MonteCarloController.MonteCarloController()
    mc.setScenarioName(scenario_name)
    mc.setNumProcesses(num_processes)
    mc.setInitialConditionsDir(ic_dir)
    mc.setArchiveDir(archive_dir)
    if execution_count is not None:
        mc.setExecutionCount(execution_count)
    mc.setShouldDisperseSeeds(disperse_seeds)
    mc.setShouldArchiveParameters(archive_parameters)

    # Extract subset of runs if run_numbers is specified
    if run_numbers is not None:
        mc.extractSubsetOfRuns(run_numbers)

    # Add retention policies if specified
    if retention_policies is not None:
        for policy in retention_policies:
            if isinstance(policy, RetentionPolicy):
                mc.addRetentionPolicy(policy)
            elif callable(policy):
                mc.addRetentionFunction(policy)
            else:
                raise ValueError("Retention policy must be a RetentionPolicy object or a callable.")

    # Run initial conditions
    mc.runInitialConditions()

if __name__ == "__main__":
    # Example usage
    rerun_monte_carlo_sims(
        scenario_name="defaultScenario",
        num_processes=2,
        run_numbers=[1, 2, 3],
        retention_policies=None,
        ic_dir="initialConditions",
        archive_dir="archive",
        execution_count=3,
        disperse_seeds=True,
        archive_parameters=True
    )
```