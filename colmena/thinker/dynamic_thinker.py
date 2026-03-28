import threading
from collections import deque
from enum import Enum
from typing import Callable, Dict, Any, List

from colmena.thinker import BaseThinker, result_processor, event_responder
from colmena.models import Result
from parsl.executors import HighThroughputExecutor


class AIState(Enum):
    """Discrete operational states for the AI component."""
    DORMANT = 1
    ACTIVE = 2


class DynamicAIThinker(BaseThinker):
    def __init__(
        self, 
        queue, 
        executors: Dict[str, HighThroughputExecutor], 
        policy_func: Callable[[List[Dict[str, Any]]], AIState],
        window_size: int = 3,
        **kwargs
    ):
        """
        Args:
            queue: Colmena client queues.
            executors: Dictionary mapping topic names (e.g., 'train', 'simulation') to Parsl executors.
            policy_func: Function evaluating the telemetry window, returning an AIState enum.
            window_size: Number of recent telemetry points to keep in the sliding window W_t.
        """
        super().__init__(queue, **kwargs)
        
        self.executors = executors
        self.policy_func = policy_func
        
        self.telemetry_window = deque(maxlen=window_size)
        
        # Initialize the state using the Enum
        self.current_ai_state = AIState.ACTIVE
        
        self.pause_ai_event = threading.Event()
        self.resume_ai_event = threading.Event()

    # -----------------------------------------------------------------------
    # Telemetry Ingestion and Policy Evaluation
    # -----------------------------------------------------------------------
    @result_processor(topic='train')
    def signal_monitor(self, result: Result):
        """Wakes up after a training result to evaluate the sliding window."""
        if not result.success:
            self.logger.warning("Training task failed, skipping telemetry evaluation.")
            return

        telemetry = result.task_info.get('telemetry')
        if telemetry is None:
            self.logger.warning("No telemetry found in task_info, skipping evaluation.")
            return

        self.telemetry_window.append(telemetry)

        if len(self.telemetry_window) < self.telemetry_window.maxlen:
            return

        target_state = self.policy_func(list(self.telemetry_window))
        self.logger.info(f"Signal Monitor evaluated target state: {target_state.name}")

        if target_state != self.current_ai_state:
            
            if target_state == AIState.DORMANT:
                self.logger.info("State mismatch: Transitioning AI to DORMANT.")
                self.current_ai_state = AIState.DORMANT
                self.pause_ai_event.set()
                
            elif target_state == AIState.ACTIVE:
                self.logger.info("State mismatch: Transitioning AI to ACTIVE.")
                self.current_ai_state = AIState.ACTIVE
                self.resume_ai_event.set()

    # -----------------------------------------------------------------------
    # Graceful Resource Reclamation
    # -----------------------------------------------------------------------
    @event_responder(event_name='pause_ai_event')
    def reclaim_resources(self):
        """Shifts resources from Train to Simulation."""
        self.pause_ai_event.clear()
        
        train_exec = self.executors.get('train')
        sim_exec = self.executors.get('simulation')
        
        if not train_exec or not sim_exec:
            self.logger.error("Executors not found in dictionary. Cannot reallocate.")
            return

        self.logger.info("Initiating graceful resource reclamation: Train -> Simulation")
        
        try:
            train_exec.scale_in(blocks=1)
            sim_exec.scale_out(blocks=1)
            self.logger.info("Successfully shifted 1 block from Train to Simulation.")
        except Exception as e:
            self.logger.error(f"Failed to reallocate resources: {e}")

    @event_responder(event_name='resume_ai_event')
    def restore_resources(self):
        """Shifts resources from Simulation back to Train."""
        self.resume_ai_event.clear()
        
        train_exec = self.executors.get('train')
        sim_exec = self.executors.get('simulation')

        if not train_exec or not sim_exec:
            return

        self.logger.info("Initiating graceful resource restoration: Simulation -> Train")
        
        try:
            sim_exec.scale_in(blocks=1)
            train_exec.scale_out(blocks=1)
            self.logger.info("Successfully shifted 1 block from Simulation to Train.")
        except Exception as e:
            self.logger.error(f"Failed to restore resources: {e}")