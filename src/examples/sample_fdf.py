from mqt.bench import get_benchmark, BenchmarkLevel
import simpy.core as sp
from src.qschedulers.cloud.qnode import QuantumNode
from src.qschedulers.cloud.qtask import QuantumTask
from src.qschedulers.schedulers.fdf import FDFScheduler
from src.qschedulers.cloud.orchestrator import Orchestrator

if __name__ == "__main__":
    from qiskit_ibm_runtime.fake_provider import FakeHanoiV2, FakeBrisbane

    # Environment
    env = sp.Environment()

    # Backends wrapped in QNodes
    qnodes = [
        QuantumNode(env, FakeHanoiV2(), name="Hanoi"),
        QuantumNode(env, FakeBrisbane(), name="Brisbane"),
    ]

    # Example tasks (circuits with different arrival times)
    tasks = [
        QuantumTask(
            id=0,
            circuit=get_benchmark("ghz", level=BenchmarkLevel.ALG, circuit_size=5),
            arrival_time=0,
        ),
        QuantumTask(
            id=1,
            circuit=get_benchmark("qft", level=BenchmarkLevel.ALG, circuit_size=10),
            arrival_time=1,
        ),
        QuantumTask(
            id=2,
            circuit=get_benchmark("ghz", level=BenchmarkLevel.ALG, circuit_size=30),
            arrival_time=1,
        ),
        QuantumTask(
            id=3,
            circuit=get_benchmark("qft", level=BenchmarkLevel.ALG, circuit_size=5),
            arrival_time=5,
        ),
    ]

    # Scheduler
    scheduler = FDFScheduler()
    orch = Orchestrator(env, scheduler, qnodes)

    orch.submit(tasks)

    # Run simulation
    env.run()

    results = orch.get_results()

    # Show results
    for r in results:
        print(r)
