"""
Example demonstrating the queue-based orchestrator with batch processing.
"""

import simpy
from qiskit import QuantumCircuit
import random
import time

from src.qschedulers.cloud.orchestrator import Orchestrator
from src.qschedulers.cloud.qnode import QuantumNode
from src.qschedulers.cloud.qtask import QuantumTask
from src.qschedulers.schedulers.round_robin import RoundRobinScheduler
from src.logger_config import setup_logger
from qiskit_ibm_runtime.fake_provider import FakeHanoiV2, FakeBrisbane

logger = setup_logger()

def create_sample_circuit(num_qubits: int = 3) -> QuantumCircuit:
    """Create a sample quantum circuit for testing."""
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc

def main():
    # Initialize simulation environment
    env = simpy.Environment()
    
    # Create quantum nodes (backends)
    qnodes = [
        QuantumNode(env, FakeHanoiV2(), name="Hanoi"),
        QuantumNode(env, FakeBrisbane(), name="Brisbane"),
    ]
    
    # Initialize scheduler and orchestrator
    scheduler = RoundRobinScheduler()
    orchestrator = Orchestrator(
        env=env,
        scheduler=scheduler,
        qnodes=qnodes,
        batch_size=3,  # Process 5 tasks at a time
        schedule_interval=10.0  # Schedule every 10 seconds
    )
    
    # Create some sample tasks with different arrival times
    tasks = []
    for i in range(20):
        circuit = create_sample_circuit()
        # Tasks arrive randomly between 0 and 50 time units
        arrival_time = random.uniform(0, 50)
        task = QuantumTask(f"task_{i}", circuit, arrival_time)
        tasks.append(task)
    
    logger.info(f"Created {len(tasks)} sample tasks")
    
    # Submit tasks to the orchestrator
    orchestrator.submit(tasks)
    
    # Run the simulation for 100 time units
    env.run(until=100)
    
    # Print results
    logger.info("\nSimulation Results:")
    completed = sum(1 for r in orchestrator.results if r["status"] == "success")
    failed = sum(1 for r in orchestrator.results if r["status"] == "failed")
    logger.info(f"Total tasks: {len(tasks)}")
    logger.info(f"Completed tasks: {completed}")
    logger.info(f"Failed tasks: {failed}")
    
    # Calculate average waiting and turnaround times for completed tasks
    completed_results = [r for r in orchestrator.results if r["status"] == "success"]
    if completed_results:
        avg_waiting = sum(r["waiting_time"] for r in completed_results) / len(completed_results)
        avg_turnaround = sum(r["turnaround_time"] for r in completed_results) / len(completed_results)
        logger.info(f"Average waiting time: {avg_waiting:.2f}")
        logger.info(f"Average turnaround time: {avg_turnaround:.2f}")

if __name__ == "__main__":
    main()