"""
APEX DIRECTOR Backend Manager
Unified interface for multiple image generators with automatic fallback cascade
"""

import asyncio
import time
import random
from typing import Dict, List, Optional, Any, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import base64

from .config import get_config, BackendConfig
from ..schemas import BACKEND_RESPONSE_SCHEMA, BACKEND_STATUS_SCHEMA

logger = logging.getLogger(__name__)


@dataclass
class BackendStatus:
    """Represents the real-time status of a backend service.

    Attributes:
        name: The name of the backend.
        status: The current status of the backend (e.g., "online", "offline").
        last_health_check: Timestamp of the last health check.
        response_time_ms: The average response time in milliseconds.
        success_rate: The success rate of the backend (0.0 to 1.0).
        queue_length: The number of jobs currently in the backend's queue.
        rate_limit_remaining: The number of requests remaining in the current
            rate limit window.
        error_count_24h: The number of errors in the last 24 hours.
        current_load: The current load of the backend (0.0 to 1.0).
        last_error: A description of the last error, if any.
        capabilities: A list of capabilities supported by the backend.
    """
    name: str
    status: str = "offline"  # online, offline, maintenance, overloaded
    last_health_check: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: int = 0
    success_rate: float = 1.0
    queue_length: int = 0
    rate_limit_remaining: int = 60
    error_count_24h: int = 0
    current_load: float = 0.0
    last_error: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the BackendStatus to a dictionary.

        Returns:
            A dictionary representation of the backend status.
        """
        data = {
            "name": self.name,
            "status": self.status,
            "last_health_check": self.last_health_check.isoformat(),
            "response_time_ms": self.response_time_ms,
            "success_rate": self.success_rate,
            "queue_length": self.queue_length,
            "rate_limit_remaining": self.rate_limit_remaining,
            "error_count_24h": self.error_count_24h,
            "current_load": self.current_load,
            "last_error": self.last_error,
            "capabilities": self.capabilities
        }
        return data


@dataclass
class GenerationRequest:
    """Represents a request for image generation.

    Attributes:
        job_id: The unique ID of the job this request belongs to.
        prompt: The text prompt for image generation.
        negative_prompt: An optional text prompt to guide the generation
            away from certain concepts.
        width: The width of the desired image.
        height: The height of the desired image.
        steps: The number of generation steps.
        guidance_scale: The guidance scale for the generation process.
        seed: An optional seed for reproducibility.
        model_variant: An optional model variant to use.
        style_preset: An optional style preset to apply.
        quality_level: The desired quality level (1-5).
        timeout: The timeout in seconds for this request.
        metadata: A dictionary for arbitrary metadata.
    """
    job_id: str
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    model_variant: Optional[str] = None
    style_preset: Optional[str] = None
    quality_level: int = 3
    timeout: int = 300
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResponse:
    """Represents the response from a generation backend.

    Attributes:
        success: True if the generation was successful, False otherwise.
        image_data: The generated image data, typically Base64 encoded.
        image_url: A URL to the generated image, if applicable.
        backend_used: The name of the backend that processed the request.
        generation_time: The time in seconds it took to generate the image.
        cost: The cost of the generation.
        error: A description of the error, if any.
        error_code: A specific error code, if any.
        metadata: A dictionary for arbitrary metadata from the backend.
    """
    success: bool
    image_data: Optional[str] = None  # Base64 encoded or URL
    image_url: Optional[str] = None
    backend_used: str = ""
    generation_time: float = 0.0
    cost: float = 0.0
    error: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackendInterface(ABC):
    """An abstract base class for backend image generation services.

    All backend implementations must inherit from this class and implement
    its abstract methods.
    """
    
    @abstractmethod
    async def generate_image(self, request: GenerationRequest) -> GenerationResponse:
        """Generates an image based on the provided request.

        Args:
            request: The GenerationRequest containing the generation
                parameters.

        Returns:
            A GenerationResponse with the result of the generation.
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> BackendStatus:
        """Performs a health check on the backend.

        Returns:
            A BackendStatus object with the current status of the backend.
        """
        pass
    
    @abstracte
    def get_capabilities(self) -> List[str]:
        """Gets a list of the backend's capabilities.

        Returns:
            A list of strings representing the backend's capabilities.
        """
        pass
    
    @abstractmethod
    def get_estimated_cost(self, request: GenerationRequest) -> float:
        """Gets the estimated cost for a given generation request.

        Args:
            request: The GenerationRequest to estimate the cost for.

        Returns:
            The estimated cost as a float.
        """
        pass


class MockBackend(BackendInterface):
    """A mock backend for testing and development purposes.

    This class simulates the behavior of a real backend, including processing
    times, occasional failures, and varying performance based on quality.
    """
    
    def __init__(self, name: str, config: BackendConfig):
        """Initializes the MockBackend.

        Args:
            name: The name of the mock backend.
            config: The configuration for the mock backend.
        """
        self.name = name
        self.config = config
        self.status = BackendStatus(
            name=name,
            status="online",
            capabilities=config.capabilities,
            rate_limit_remaining=config.rate_limit
        )
        # Simulate varying performance based on quality level
        self.performance_factors = {
            1: 0.3,  # Fast but low quality
            2: 0.5,
            3: 1.0,  # Baseline
            4: 1.5,
            5: 2.5   # Slow but high quality
        }
    
    async def generate_image(self, request: GenerationRequest) -> GenerationResponse:
        """Simulates generating an image.

        This method simulates processing time, occasional failures, and returns
        a mock response.

        Args:
            request: The GenerationRequest containing the generation
                parameters.

        Returns:
            A GenerationResponse with the simulated result.
        """
        start_time = time.time()
        
        try:
            # Simulate processing time based on quality level
            base_time = self.performance_factors.get(request.quality_level, 1.0)
            processing_time = base_time * random.uniform(2, 8)
            
            # Simulate occasional failures (higher for lower quality backends)
            if random.random() < (0.05 / request.quality_level):
                await asyncio.sleep(processing_time)
                return GenerationResponse(
                    success=False,
                    backend_used=self.name,
                    error=f"Simulated {self.name} failure",
                    error_code="SERVICE_ERROR"
                )
            
            await asyncio.sleep(processing_time)
            
            # Generate mock image data
            image_data = f"mock_image_data_{self.name}_{request.job_id}"
            encoded_data = base64.b64encode(image_data.encode()).decode()
            
            generation_time = time.time() - start_time
            cost = self.get_estimated_cost(request)
            
            return GenerationResponse(
                success=True,
                image_data=encoded_data,
                backend_used=self.name,
                generation_time=generation_time,
                cost=cost,
                metadata={
                    "model_used": f"{self.name}_v1.0",
                    "steps_used": request.steps,
                    "seed": request.seed or random.randint(0, 999999),
                    "quality_metrics": {
                        "sharpness": random.uniform(6, 10),
                        "contrast": random.uniform(6, 10),
                        "color_balance": random.uniform(6, 10)
                    }
                }
            )
            
        except Exception as e:
            return GenerationResponse(
                success=False,
                backend_used=self.name,
                error=str(e),
                error_code="GENERATION_ERROR"
            )
    
    async def health_check(self) -> BackendStatus:
        """Simulates a health check.

        This method simulates occasional offline status and fluctuating
        performance metrics.

        Returns:
            A BackendStatus object with the simulated status.
        """
        # Simulate occasional offline status
        if random.random() < 0.02:  # 2% chance of being offline
            self.status.status = "offline"
            self.status.last_error = "Service unavailable"
        else:
            self.status.status = "online"
            self.status.last_error = None
        
        self.status.last_health_check = datetime.utcnow()
        self.status.response_time_ms = random.randint(50, 500)
        self.status.success_rate = random.uniform(0.9, 0.99)
        self.status.queue_length = random.randint(0, 10)
        self.status.rate_limit_remaining = max(0, self.status.rate_limit_remaining - random.randint(0, 5))
        self.status.current_load = random.uniform(0.1, 0.8)
        
        return self.status
    
    def get_capabilities(self) -> List[str]:
        """Gets the capabilities of the mock backend.

        Returns:
            A list of strings representing the backend's capabilities.
        """
        return self.config.capabilities.copy()
    
    def get_estimated_cost(self, request: GenerationRequest) -> float:
        """Gets the estimated cost for a given request.

        The cost is calculated based on the base cost, image size, and
        quality level.

        Args:
            request: The GenerationRequest to estimate the cost for.

        Returns:
            The estimated cost.
        """
        base_cost = self.config.cost_per_image
        # Adjust cost based on image size and quality
        size_multiplier = (request.width * request.height) / (512 * 512)
        quality_multiplier = request.quality_level / 3.0
        return base_cost * size_multiplier * quality_multiplier


class BackendManager:
    """Manages multiple backend services with automatic fallback.

    This class is responsible for initializing, monitoring, and selecting
    appropriate backends for generation requests. It also handles fallback
    logic when a primary backend fails.
    """
    
    def __init__(self, custom_configs: Optional[List[BackendConfig]] = None):
        """Initializes the BackendManager.

        Args:
            custom_configs: An optional list of BackendConfig to override the
                default configuration.
        """
        self.config_manager = get_config()
        self.configs = custom_configs or self.config_manager.get_backend_configs()
        self.backends: Dict[str, BackendInterface] = {}
        self.status_tracker: Dict[str, BackendStatus] = {}
        self.fallback_chain: List[str] = []
        self.health_check_tasks: List[asyncio.Task] = []
        
        # Initialize backends
        self._initialize_backends()
        
        # Start health monitoring
        self._start_health_monitoring()
    
    def _initialize_backends(self):
        """Initializes backend instances based on the loaded configuration.

        This method creates and registers backend instances and builds the
        fallback chain.
        """
        for config in self.configs:
            if config.enabled:
                try:
                    backend = self._create_backend_instance(config)
                    if backend:
                        self.backends[config.name] = backend
                        self.status_tracker[config.name] = BackendStatus(
                            name=config.name,
                            capabilities=config.capabilities
                        )
                        logger.info(f"Initialized backend: {config.name}")
                except Exception as e:
                    logger.error(f"Failed to initialize backend {config.name}: {e}")
        
        # Build fallback chain based on priority
        self.fallback_chain = [config.name for config in self.configs if config.enabled]
        logger.info(f"Fallback chain: {self.fallback_chain}")
    
    def _create_backend_instance(self, config: BackendConfig) -> Optional[BackendInterface]:
        """Creates a backend instance from a BackendConfig.

        In a production environment, this method would instantiate real
        backend clients. For now, it creates MockBackend instances.

        Args:
            config: The configuration for the backend to create.

        Returns:
            An instance of a BackendInterface, or None if the backend
            type is unknown.
        """
        # For this implementation, we'll use MockBackend for all backends
        # In production, this would create real backend instances
        
        if "nano_banana" in config.name.lower():
            return MockBackend("Nano Banana", config)
        elif "imagen" in config.name.lower():
            return MockBackend("Google Imagen", config)
        elif "minimax" in config.name.lower():
            return MockBackend("MiniMax", config)
        elif "sdxl" in config.name.lower() or "stable" in config.name.lower():
            return MockBackend("SDXL", config)
        else:
            return MockBackend(config.name, config)
    
    def _start_health_monitoring(self):
        """Starts the background health monitoring tasks for each backend."""
        for backend_name in self.backends.keys():
            task = asyncio.create_task(self._health_monitor_loop(backend_name))
            self.health_check_tasks.append(task)
    
    async def _health_monitor_loop(self, backend_name: str):
        """The background loop for monitoring the health of a single backend.

        Args:
            backend_name: The name of the backend to monitor.
        """
        while True:
            try:
                backend = self.backends.get(backend_name)
                if backend:
                    status = await backend.health_check()
                    self.status_tracker[backend_name] = status
                    
                    # Log status changes
                    if status.status != "online" and status.last_error:
                        logger.warning(f"Backend {backend_name} status: {status.status} - {status.last_error}")
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Health check failed for {backend_name}: {e}")
                await asyncio.sleep(30)  # Retry sooner on error
    
    async def generate_with_fallback(self, request: GenerationRequest) -> GenerationResponse:
        """Generates an image with automatic fallback to backup backends.

        This method iterates through the fallback chain, attempting to
        generate the image with each backend until one succeeds.

        Args:
            request: The GenerationRequest for the image.

        Returns:
            A GenerationResponse, either successful or with an error indicating
            that all backends failed.
        """
        attempted_backends = []
        
        # Try each backend in the fallback chain
        for backend_name in self.fallback_chain:
            if backend_name in attempted_backends:
                continue
            
            backend = self.backends.get(backend_name)
            if not backend:
                continue
            
            status = self.status_tracker.get(backend_name)
            
            # Skip offline or overloaded backends
            if status and status.status in ["offline", "maintenance"]:
                logger.info(f"Skipping backend {backend_name}: status {status.status}")
                continue
            
            # Check rate limits
            if status and status.rate_limit_remaining <= 0:
                logger.info(f"Skipping backend {backend_name}: rate limit exceeded")
                continue
            
            # Check success rate
            if status and status.success_rate < 0.8:
                logger.warning(f"Backend {backend_name} has low success rate: {status.success_rate}")
            
            logger.info(f"Attempting generation with {backend_name}")
            attempted_backends.append(backend_name)
            
            try:
                response = await asyncio.wait_for(
                    backend.generate_image(request),
                    timeout=request.timeout
                )
                
                if response.success:
                    logger.info(f"Generation successful with {backend_name}")
                    return response
                else:
                    logger.warning(f"Generation failed with {backend_name}: {response.error}")
                    continue
                    
            except asyncio.TimeoutError:
                logger.warning(f"Generation timeout with {backend_name}")
                continue
            except Exception as e:
                logger.error(f"Generation error with {backend_name}: {e}")
                continue
        
        # All backends failed
        return GenerationResponse(
            success=False,
            error="All backends failed to generate image",
            error_code="ALL_BACKENDS_FAILED"
        )
    
    async def generate_parallel(self, requests: List[GenerationRequest]) -> List[GenerationResponse]:
        """Generates multiple images in parallel using available backends.

        This method distributes the generation requests across available
        backends and processes them concurrently.

        Args:
            requests: A list of GenerationRequest objects.

        Returns:
            A list of GenerationResponse objects corresponding to the
            requests.
        """
        if not requests:
            return []
        
        # Distribute requests across available backends
        responses = []
        tasks = []
        
        for request in requests:
            task = asyncio.create_task(self.generate_with_fallback(request))
            tasks.append((request.job_id, task))
        
        # Wait for all generations to complete
        for job_id, task in tasks:
            try:
                response = await task
                responses.append(response)
            except Exception as e:
                logger.error(f"Parallel generation failed for {job_id}: {e}")
                responses.append(GenerationResponse(
                    success=False,
                    error=str(e),
                    error_code="PARALLEL_ERROR"
                ))
        
        return responses
    
    def get_backend_status(self, backend_name: str) -> Optional[BackendStatus]:
        """Gets the current status of a specific backend.

        Args:
            backend_name: The name of the backend to get the status for.

        Returns:
            A BackendStatus object, or None if the backend is not found.
        """
        return self.status_tracker.get(backend_name)
    
    def get_all_backend_status(self) -> Dict[str, BackendStatus]:
        """Gets the status of all managed backends.

        Returns:
            A dictionary mapping backend names to their BackendStatus objects.
        """
        return self.status_tracker.copy()
    
    def get_best_backend(self, requirements: Dict[str, Any]) -> Optional[str]:
        """Selects the best backend based on a set of requirements.

        This method scores available backends based on factors like quality,
        cost, and current load.

        Args:
            requirements: A dictionary of requirements, such as "min_quality"
                or "max_cost".

        Returns:
            The name of the best backend, or None if no suitable backend is
            found.
        """
        # Score backends based on requirements
        scores = {}
        
        for backend_name in self.fallback_chain:
            backend = self.backends.get(backend_name)
            status = self.status_tracker.get(backend_name)
            
            if not backend or not status:
                continue
            
            # Skip offline backends
            if status.status != "online":
                continue
            
            score = 100  # Base score
            
            # Quality requirement
            if "min_quality" in requirements:
                config = next((c for c in self.configs if c.name == backend_name), None)
                if config and config.quality_level < requirements["min_quality"]:
                    continue  # Skip if quality too low
            
            # Cost optimization
            if "max_cost" in requirements:
                # Would need to calculate estimated cost
                pass
            
            # Load balancing
            score -= int(status.current_load * 50)
            
            # Success rate bonus
            score += int(status.success_rate * 20)
            
            scores[backend_name] = score
        
        # Return best scored backend
        if scores:
            return max(scores, key=scores.get)
        
        return None
    
    async def benchmark_backends(self, test_request: GenerationRequest) -> Dict[str, Dict[str, Any]]:
        """Benchmarks all available backends with a test request.

        Args:
            test_request: The GenerationRequest to use for benchmarking.

        Returns:
            A dictionary with the benchmark results for each backend.
        """
        results = {}
        
        for backend_name in self.backends.keys():
            status = self.status_tracker.get(backend_name)
            if not status or status.status != "online":
                continue
            
            logger.info(f"Benchmarking {backend_name}")
            
            # Create test request for this backend
            test_req = GenerationRequest(
                job_id=f"benchmark_{backend_name}",
                prompt=test_request.prompt,
                width=test_request.width,
                height=test_request.height,
                quality_level=test_request.quality_level,
                timeout=60  # Shorter timeout for benchmarks
            )
            
            try:
                start_time = time.time()
                response = await asyncio.wait_for(
                    self.backends[backend_name].generate_image(test_req),
                    timeout=60
                )
                end_time = time.time()
                
                results[backend_name] = {
                    "success": response.success,
                    "generation_time": response.generation_time,
                    "total_time": end_time - start_time,
                    "cost": response.cost,
                    "quality_score": response.metadata.get("quality_metrics", {}),
                    "error": response.error if not response.success else None
                }
                
            except Exception as e:
                results[backend_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def update_backend_config(self, backend_name: str, updates: Dict[str, Any]):
        """Updates the configuration for a specific backend.

        Args:
            backend_name: The name of the backend to update.
            updates: A dictionary of configuration updates.
        """
        self.config_manager.update_backend_config(backend_name, updates)
        
        # Update local config if exists
        for config in self.configs:
            if config.name == backend_name:
                for key, value in updates.items():
                    setattr(config, key, value)
                break
    
    def enable_backend(self, backend_name: str, enabled: bool = True):
        """Enables or disables a backend.

        Args:
            backend_name: The name of the backend.
            enabled: True to enable the backend, False to disable it.
        """
        self.config_manager.enable_backend(backend_name, enabled)
        
        if enabled and backend_name not in self.backends:
            # Re-initialize the backend
            config = next((c for c in self.configs if c.name == backend_name), None)
            if config:
                backend = self._create_backend_instance(config)
                if backend:
                    self.backends[backend_name] = backend
                    self.fallback_chain.insert(config.priority, backend_name)
        elif not enabled and backend_name in self.backends:
            # Disable the backend
            if backend_name in self.fallback_chain:
                self.fallback_chain.remove(backend_name)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Gets overall statistics for the backend manager.

        Returns:
            A dictionary of statistics, including the number of online
            backends, average response time, and success rate.
        """
        total_backends = len(self.backends)
        online_backends = sum(1 for status in self.status_tracker.values() if status.status == "online")
        
        avg_response_time = 0
        avg_success_rate = 0
        total_queue = 0
        
        if self.status_tracker:
            response_times = [status.response_time_ms for status in self.status_tracker.values()]
            success_rates = [status.success_rate for status in self.status_tracker.values()]
            queue_lengths = [status.queue_length for status in self.status_tracker.values()]
            
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
            total_queue = sum(queue_lengths)
        
        return {
            "total_backends": total_backends,
            "online_backends": online_backends,
            "availability_percent": (online_backends / total_backends * 100) if total_backends > 0 else 0,
            "average_response_time_ms": avg_response_time,
            "average_success_rate": avg_success_rate,
            "total_queue_length": total_queue,
            "fallback_chain": self.fallback_chain
        }
    
    async def shutdown(self):
        """Gracefully shuts down the backend manager.

        This method cancels all background health monitoring tasks.
        """
        # Cancel health monitoring tasks
        for task in self.health_check_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.health_check_tasks:
            await asyncio.gather(*self.health_check_tasks, return_exceptions=True)
        
        logger.info("Backend manager shutdown complete")


# Global backend manager instance
_backend_manager: Optional[BackendManager] = None


def get_backend_manager() -> BackendManager:
    """Gets the global instance of the BackendManager.

    This function implements a singleton pattern to ensure that only one
    instance of the backend manager exists.

    Returns:
        The global BackendManager instance.
    """
    global _backend_manager
    if _backend_manager is None:
        _backend_manager = BackendManager()
    return _backend_manager