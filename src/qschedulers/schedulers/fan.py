from typing import Any
from src.logger_config import setup_logger
from typing import Any
from .base import Scheduler

logger = setup_logger()

class FANScheduler(Scheduler):
    """
    First-Available-None (FAN) Scheduler.
    Assigns each QTask to the first idle quantum node to minimize waiting time.
    """

    def __init__(self):
        logger.info("Initialized FANScheduler.")

    def schedule(self, tasks: list[Any], qnodes: list[Any]) -> dict[str, Any]:
        logger.info(f"Scheduling {len(tasks)} tasks across {len(qnodes)} qnodes using FAN policy.")
        if not qnodes:
            logger.error("No quantum nodes provided for scheduling.")
            raise ValueError("No quantum nodes provided for scheduling.")

        # Track next available time for each qnode
        qnode_available = {qnode: 0.0 for qnode in qnodes}
        assignments = []

        for task_id, task in enumerate(tasks):
            # Find qnode with earliest available time
            earliest_qnode = min(qnodes, key=lambda q: qnode_available[q])
            start_time = qnode_available[earliest_qnode]

            # Estimate task duration if available, else assume 1.0
            duration = getattr(task, "estimated_duration", 1.0)
            qnode_available[earliest_qnode] += duration

            logger.debug(f"Assigning task {task_id} to qnode {getattr(earliest_qnode, 'name', earliest_qnode)} at time {start_time} (duration={duration})")
            assignments.append((task_id, earliest_qnode))

        logger.info(f"Completed scheduling. Assignments: {assignments}")
        return {
            "assignments": assignments,
            "metadata": {
                "policy": "first_available_none",
                "num_tasks": len(tasks),
                "num_qnodes": len(qnodes),
            },
        }
