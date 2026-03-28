from proxystore.proxy import Proxy
from proxystore.store import get_store

# Global registry acting as persistent VRAM for the worker process
_GPU_VRAM_REGISTRY = {}

class DummyModelState:
    """A dummy struct representing the model weights and optimizer state."""
    def __init__(self, weights):
        self.weights = weights
        self.is_on_gpu = False

    def to_gpu(self):
        self.is_on_gpu = True
        
    def to_cpu(self):
        self.is_on_gpu = False

def gpu_startup(redis_proxy: Proxy = None) -> str:
    """
    State 1 -> 2: Fetches state from Redis (if exists), allocates on GPU, 
    and returns a reference pointer.
    """
    if redis_proxy is not None:
        # Resolve the proxy to pull the state byte stream from Redis
        # Deserialization happens automatically here
        model_state = redis_proxy 
    else:
        # First time initialization
        model_state = DummyModelState(weights=[0.1, 0.2, 0.3])

    # "Allocate" on GPU
    model_state.to_gpu()
    
    # Generate a unique pointer ID and store it in the worker's persistent memory
    device_pointer = f"gpu_ptr_{id(model_state)}"
    _GPU_VRAM_REGISTRY[device_pointer] = model_state
    
    return device_pointer

def gpu_train(device_pointer: str, training_data: list) -> dict:
    """
    Active State 2: Uses the pointer to access the resident model and train.
    """
    if device_pointer not in _GPU_VRAM_REGISTRY:
        raise MemoryError(f"Segmentation Fault: {device_pointer} not found in VRAM.")
        
    # Dereference the pointer
    model = _GPU_VRAM_REGISTRY[device_pointer]
    
    # Execute training (in-place modification)
    model.weights = [w + 0.01 for w in model.weights]
    
    return {"loss": 0.05, "telemetry_value": 0.99}

def gpu_teardown(device_pointer: str) -> Proxy:
    """
    State 2 -> 1: Extracts model, pushes to Redis, frees VRAM.
    """
    if device_pointer not in _GPU_VRAM_REGISTRY:
        raise MemoryError("Cannot teardown: Pointer invalid.")

    # 1. Extract from VRAM and move to host memory
    model = _GPU_VRAM_REGISTRY.pop(device_pointer)
    model.to_cpu()
    
    # 2. Serialize and push to Redis via ProxyStore
    # This offloads the heavy data to the Redis cluster and returns a lightweight reference
    store = get_store('file') # Or 'redis', assuming configured in __main__
    redis_proxy = store.proxy(model)
    
    # 3. Memory is now deallocated from _GPU_VRAM_REGISTRY
    return redis_proxy