```python
import os
from Basilisk.simulation import SimulationExecutor, InitializeSimulation, ExecuteSimulation
from Basilisk.utilities import test_MonteCarloSimulation, setSimulationFunction, reRunCases, runInitialConditions
from Basilisk.utilities import VariableRetentionParameters, addRetentionPoliciesToSim

def rerun_monte_carlo_simulations(
    scenario_name,
    num_processes=1,
    run_numbers=None,
    retention_policies=None,
    initial_conditions_dir="initialConditions",
    archive_dir="archive",
    execution_count=100,
    disperse_seeds=True,
    archive_parameters=True
):
    # Set up Monte Carlo controller
    mcSim = test_MonteCarloSimulation()
    mcSim.setSimulationFunction(scenario_name)
    mcSim.setExecutionCount(execution_count)
    mcSim.setInitialConditionsDir(initial_conditions_dir)
    mcSim.setArchiveDir(archive_dir)
    mcSim.setDisperseSeeds(disperse_seeds)
    mcSim.setArchiveParameters(archive_parameters)
    mcSim.setNumProcesses(num_processes)

    # Add retention policies if provided
    if retention_policies:
        for policy in retention_policies:
            addRetentionPoliciesToSim(mcSim, policy)

    # Rerun specified cases
    if run_numbers is not None:
        reRunCases(mcSim, run_numbers)
    else:
        runInitialConditions(mcSim)

if __name__ == "__main__":
    # Example usage
    rerun_monte_carlo_simulations(
        scenario_name="test_scenarioBskMcScenarios",
        num_processes=4,
        run_numbers=[1, 2, 5],
        retention_policies=[VariableRetentionParameters("keepAll", True)]
    )
```