import threading
from collections import deque
from enum import Enum
from typing import Callable, Dict, Any, List

from colmena.thinker import BaseThinker, result_processor, event_responder
from colmena.models import Result
from parsl.executors import HighThroughputExecutor

class AIState(Enum):
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
        super().__init__(queue, **kwargs)
        self.executors = executors
        self.policy_func = policy_func
        self.telemetry_window = deque(maxlen=window_size)
        
        self.current_ai_state = AIState.DORMANT # Start dormant to trigger initial startup
        
        self.pause_ai_event = threading.Event()
        self.resume_ai_event = threading.Event()
        
        # --- Memory Management Pointers ---
        self.active_device_pointer = None
        self.redis_state_proxy = None 
        
        # Trigger initial boot sequence
        self.resume_ai_event.set()

    # -----------------------------------------------------------------------
    # Memory Allocation & Resource Scaling
    # -----------------------------------------------------------------------
    @event_responder(event_name='resume_ai_event')
    def boot_and_allocate(self):
        """State 1 -> 2: Allocate hardware, then submit startup task."""
        self.resume_ai_event.clear()
        self.logger.info("Allocating hardware for AI...")
        
        # 1. Hardware scaling
        self.executors['simulation'].scale_in(blocks=1)
        self.executors['train'].scale_out(blocks=1)
        
        # 2. Submit startup task to load model from Redis to GPU VRAM
        self.logger.info("Hardware allocated. Submitting GPU startup task.")
        self.queues.send_inputs(
            self.redis_state_proxy, 
            method='gpu_startup', 
            topic='startup'
        )

    @result_processor(topic='startup')
    def process_startup(self, result: Result):
        """Receives the device pointer once allocation is complete."""
        if result.success:
            self.active_device_pointer = result.value
            self.current_ai_state = AIState.ACTIVE
            self.logger.info(f"Model successfully loaded. Device pointer: {self.active_device_pointer}")
            # The Thinker is now ready to submit 'train' tasks using this pointer.
        else:
            self.logger.error(f"Startup failed: {result.exc_info}")

    # -----------------------------------------------------------------------
    # Training Execution & Policy Evaluation
    # -----------------------------------------------------------------------
    @result_processor(topic='train')
    def signal_monitor(self, result: Result):
        """Evaluates policy after training."""
        if not result.success:
            return

        self.telemetry_window.append(result.task_info.get('telemetry', {}))

        if len(self.telemetry_window) < self.telemetry_window.maxlen:
            return

        target_state = self.policy_func(list(self.telemetry_window))

        if target_state == AIState.DORMANT and self.current_ai_state == AIState.ACTIVE:
            self.logger.info("Convergence detected. Initiating teardown.")
            self.current_ai_state = AIState.DORMANT
            self.pause_ai_event.set()

    # -----------------------------------------------------------------------
    # Memory Deallocation & Hardware Reclamation
    # -----------------------------------------------------------------------
    @event_responder(event_name='pause_ai_event')
    def initiate_teardown(self):
        """State 2 -> 1: Submit teardown task before yielding hardware."""
        self.pause_ai_event.clear()
        
        self.logger.info(f"Evicting model at {self.active_device_pointer} to Redis...")
        
        # Send task to extract model from VRAM and push to Redis
        self.queues.send_inputs(
            self.active_device_pointer, 
            method='gpu_teardown', 
            topic='teardown'
        )

    @result_processor(topic='teardown')
    def finalize_reclamation(self, result: Result):
        """Receives the Redis proxy, clears the pointer, and yields hardware."""
        if result.success:
            # 1. Update pointers
            self.redis_state_proxy = result.value
            self.active_device_pointer = None
            self.logger.info("Model state safely preserved in Redis cluster.")
            
            # 2. Yield hardware back to simulation
            self.logger.info("Reclaiming GPUs for simulation workflows.")
            self.executors['train'].scale_in(blocks=1)
            self.executors['simulation'].scale_out(blocks=1)
        else:
            self.logger.error(f"Teardown failed: {result.exc_info}")