"""
Unit tests for APEX DIRECTOR Backend Manager

Tests multi-backend abstraction and management functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from apex_director.core.backend_manager import BackendManager, BackendInterface
from apex_director.core.backend_manager import (
    NanoBananaBackend, 
    GoogleImagenBackend, 
    MiniMaxBackend,
    SDXLBackend
)


class MockBackend(BackendInterface):
    """Mock backend for testing"""
    
    def __init__(self, name="mock", should_fail=False):
        self.name = name
        self.should_fail = should_fail
        self.call_count = 0
    
    async def initialize(self):
        return True
    
    async def generate_image(self, prompt, **kwargs):
        self.call_count += 1
        
        if self.should_fail:
            raise Exception(f"{self.name} backend failed")
        
        return {
            "image_path": f"/tmp/{self.name}_output.png",
            "metadata": {
                "backend": self.name,
                "prompt": prompt,
                "generated_at": "2024-01-01"
            }
        }
    
    async def health_check(self):
        return not self.should_fail
    
    async def get_capabilities(self):
        return {
            "max_resolution": (1024, 1024),
            "supported_formats": ["png", "jpg"],
            "quality_levels": [1, 2, 3, 4, 5]
        }
    
    async def estimate_cost(self, **kwargs):
        return 0.05


class TestBackendManager:
    """Test suite for BackendManager"""
    
    @pytest.fixture
    def backend_manager(self):
        """Create backend manager instance"""
        config = {
            "backends": {
                "mock1": {"enabled": True, "priority": 1},
                "mock2": {"enabled": True, "priority": 2},
                "mock3": {"enabled": False, "priority": 3}
            }
        }
        manager = BackendManager(config=config)
        return manager
    
    @pytest.fixture
    def mock_backends(self):
        """Create mock backends"""
        return {
            "mock1": MockBackend("mock1"),
            "mock2": MockBackend("mock2"),
            "mock3": MockBackend("mock3")
        }
    
    def test_initialization(self, backend_manager):
        """Test backend manager initialization"""
        assert backend_manager.config is not None
        assert len(backend_manager.available_backends) == 0
    
    @pytest.mark.asyncio
    async def test_register_backend(self, backend_manager, mock_backends):
        """Test backend registration"""
        for name, backend in mock_backends.items():
            backend_manager.register_backend(name, backend)
        
        assert len(backend_manager.available_backends) == 3
        assert "mock1" in backend_manager.available_backends
        assert "mock2" in backend_manager.available_backends
        assert "mock3" in backend_manager.available_backends
    
    @pytest.mark.asyncio
    async def test_initialize_backends(self, backend_manager, mock_backends):
        """Test backend initialization"""
        for name, backend in mock_backends.items():
            backend_manager.register_backend(name, backend)
        
        await backend_manager.initialize_backends()
        
        # All mock backends should initialize successfully
        assert backend_manager.initialized_backends["mock1"] == True
        assert backend_manager.initialized_backends["mock2"] == True
        assert backend_manager.initialized_backends["mock3"] == True
    
    @pytest.mark.asyncio
    async def test_generate_with_fallback(self, backend_manager, mock_backends):
        """Test generation with automatic fallback"""
        # Setup: one working backend, one failing backend
        mock_backends["mock1"].should_fail = True
        mock_backends["mock2"].should_fail = False
        
        backend_manager.register_backend("mock1", mock_backends["mock1"])
        backend_manager.register_backend("mock2", mock_backends["mock2"])
        
        await backend_manager.initialize_backends()
        
        # Generate should succeed with fallback
        result = await backend_manager.generate_with_fallback(
            prompt="A test prompt"
        )
        
        assert result is not None
        assert result["metadata"]["backend"] == "mock2"  # Used working backend
        assert mock_backends["mock2"].call_count == 1
    
    @pytest.mark.asyncio
    async def test_generate_all_backends(self, backend_manager, mock_backends):
        """Test generation across all backends"""
        backend_manager.register_backend("mock1", mock_backends["mock1"])
        backend_manager.register_backend("mock2", mock_backends["mock2"])
        
        await backend_manager.initialize_backends()
        
        results = await backend_manager.generate_all_backends(
            prompt="Test prompt"
        )
        
        assert len(results) == 2
        assert "mock1" in results
        assert "mock2" in results
        assert mock_backends["mock1"].call_count == 1
        assert mock_backends["mock2"].call_count == 1
    
    @pytest.mark.asyncio
    async def test_select_best_backend(self, backend_manager, mock_backends):
        """Test backend selection based on criteria"""
        backend_manager.register_backend("mock1", mock_backends["mock1"])
        backend_manager.register_backend("mock2", mock_backends["mock2"])
        
        await backend_manager.initialize_backends()
        
        # Select based on quality
        best_backend = await backend_manager.select_best_backend(
            criteria="quality"
        )
        assert best_backend == "mock1"  # Higher priority
        
        # Select based on cost
        best_backend = await backend_manager.select_best_backend(
            criteria="cost"
        )
        assert best_backend in ["mock1", "mock2"]
    
    @pytest.mark.asyncio
    async def test_health_check(self, backend_manager, mock_backends):
        """Test backend health monitoring"""
        # Setup: one healthy, one unhealthy backend
        mock_backends["mock1"].should_fail = False
        mock_backends["mock2"].should_fail = True
        
        backend_manager.register_backend("mock1", mock_backends["mock1"])
        backend_manager.register_backend("mock2", mock_backends["mock2"])
        
        await backend_manager.initialize_backends()
        
        health_status = await backend_manager.check_all_backends()
        
        assert health_status["mock1"]["healthy"] == True
        assert health_status["mock2"]["healthy"] == False
        assert health_status["mock1"]["error"] is None
        assert health_status["mock2"]["error"] is not None
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, backend_manager, mock_backends):
        """Test circuit breaker pattern for failing backends"""
        # Make backend consistently fail
        mock_backends["mock1"].should_fail = True
        
        backend_manager.register_backend("mock1", mock_backends["mock1"])
        await backend_manager.initialize_backends()
        
        # First few attempts should try the failing backend
        results = []
        for _ in range(5):
            try:
                result = await backend_manager.generate_with_fallback("test")
                results.append(result)
            except Exception:
                results.append(None)
        
        # Eventually circuit breaker should activate
        # (implementation depends on circuit breaker configuration)
        circuit_status = backend_manager.get_circuit_status("mock1")
        assert circuit_status is not None
    
    @pytest.mark.asyncio
    async def test_load_balancing(self, backend_manager, mock_backends):
        """Test load balancing across backends"""
        backend_manager.register_backend("mock1", mock_backends["mock1"])
        backend_manager.register_backend("mock2", mock_backends["mock2"])
        
        await backend_manager.initialize_backends()
        
        # Submit multiple generation requests
        for _ in range(10):
            await backend_manager.generate_with_fallback("test prompt")
        
        # Both backends should be used
        assert mock_backends["mock1"].call_count > 0
        assert mock_backends["mock2"].call_count > 0
    
    @pytest.mark.asyncio
    async def test_cost_estimation(self, backend_manager, mock_backends):
        """Test cost estimation across backends"""
        backend_manager.register_backend("mock1", mock_backends["mock1"])
        backend_manager.register_backend("mock2", mock_backends["mock2"])
        
        await backend_manager.initialize_backends()
        
        cost_estimate = await backend_manager.estimate_costs(
            params={"width": 1024, "height": 1024}
        )
        
        assert "mock1" in cost_estimate
        assert "mock2" in cost_estimate
        assert cost_estimate["mock1"] > 0
        assert cost_estimate["mock2"] > 0
    
    def test_get_backend_statistics(self, backend_manager, mock_backends):
        """Test backend usage statistics"""
        backend_manager.register_backend("mock1", mock_backends["mock1"])
        backend_manager.register_backend("mock2", mock_backends["mock2"])
        
        # Simulate some usage
        mock_backends["mock1"].call_count = 5
        mock_backends["mock2"].call_count = 3
        
        stats = backend_manager.get_statistics()
        
        assert "mock1" in stats
        assert "mock2" in stats
        assert stats["mock1"]["call_count"] == 5
        assert stats["mock2"]["call_count"] == 3
    
    @pytest.mark.asyncio
    async def test_capability_matching(self, backend_manager, mock_backends):
        """Test capability-based backend selection"""
        # Mock capabilities
        mock_backends["mock1"].get_capabilities = AsyncMock(return_value={
            "max_resolution": (512, 512),
            "supported_formats": ["png"],
            "quality_levels": [1, 2, 3]
        })
        
        mock_backends["mock2"].get_capabilities = AsyncMock(return_value={
            "max_resolution": (1024, 1024),
            "supported_formats": ["png", "jpg"],
            "quality_levels": [1, 2, 3, 4, 5]
        })
        
        backend_manager.register_backend("mock1", mock_backends["mock1"])
        backend_manager.register_backend("mock2", mock_backends["mock2"])
        
        await backend_manager.initialize_backends()
        
        # Select backend capable of 1024x1024
        suitable_backend = await backend_manager.select_backend_by_capability(
            required_resolution=(1024, 1024),
            required_format="jpg"
        )
        
        assert suitable_backend == "mock2"
    
    @pytest.mark.asyncio
    async def test_backend_timeout_handling(self, backend_manager, mock_backends):
        """Test timeout handling for slow backends"""
        # Mock a slow backend
        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(2)  # 2 second delay
            return {"image_path": "/tmp/slow.png"}
        
        mock_backends["slow"] = MockBackend("slow")
        mock_backends["slow"].generate_image = slow_generate
        
        backend_manager.register_backend("slow", mock_backends["slow"])
        await backend_manager.initialize_backends()
        
        # Generation with timeout should handle slow backend gracefully
        result = await backend_manager.generate_with_fallback(
            prompt="test",
            timeout=0.5  # 0.5 second timeout
        )
        
        # Should timeout and potentially return None or error
        assert result is None or "error" in str(result)


class TestBackendInterface:
    """Test suite for BackendInterface"""
    
    @pytest.mark.asyncio
    async def test_backend_interface_contract(self):
        """Test that backends implement the interface correctly"""
        
        class TestBackend(BackendInterface):
            async def initialize(self):
                return True
            
            async def generate_image(self, prompt, **kwargs):
                return {"path": "test.png"}
            
            async def health_check(self):
                return True
            
            async def get_capabilities(self):
                return {}
            
            async def estimate_cost(self, **kwargs):
                return 0.1
        
        backend = TestBackend()
        
        # Test all interface methods exist
        assert hasattr(backend, 'initialize')
        assert hasattr(backend, 'generate_image')
        assert hasattr(backend, 'health_check')
        assert hasattr(backend, 'get_capabilities')
        assert hasattr(backend, 'estimate_cost')
        
        # Test methods are callable
        assert callable(backend.initialize)
        assert callable(backend.generate_image)
        assert callable(backend.health_check)
        assert callable(backend.get_capabilities)
        assert callable(backend.estimate_cost)
        
        # Test methods work
        result = await backend.initialize()
        assert result == True
        
        result = await backend.generate_image("test prompt")
        assert result["path"] == "test.png"
        
        result = await backend.health_check()
        assert result == True


class TestSpecificBackends:
    """Test suite for specific backend implementations"""
    
    @pytest.mark.asyncio
    async def test_nano_banana_backend(self):
        """Test Nano Banana backend (mocked)"""
        backend = NanoBananaBackend(api_key="test_key")
        
        # Mock the actual API calls
        with patch.object(backend, '_api_call') as mock_api:
            mock_api.return_value = {"image_url": "http://example.com/image.png"}
            
            result = await backend.generate_image("A sunset")
            assert "image_path" in result
    
    @pytest.mark.asyncio
    async def test_minimax_backend(self):
        """Test MiniMax backend (mocked)"""
        backend = MiniMaxBackend(api_key="test_key", endpoint="test_endpoint")
        
        with patch.object(backend, '_api_call') as mock_api:
            mock_api.return_value = {"data": {"url": "http://example.com/minimax.png"}}
            
            result = await backend.generate_image("A mountain")
            assert "image_path" in result
    
    @pytest.mark.asyncio
    async def test_sdxl_backend(self):
        """Test SDXL backend (mocked)"""
        backend = SDXLBackend(api_key="test_key")
        
        with patch.object(backend, '_api_call') as mock_api:
            mock_api.return_value = {"images": [{"url": "http://example.com/sdxl.png"}]}
            
            result = await backend.generate_image("A forest")
            assert "image_path" in result


@pytest.mark.asyncio
async def test_backend_manager_integration():
    """Integration test for backend manager"""
    
    config = {
        "backends": {
            "test_backend": {"enabled": True, "priority": 1}
        }
    }
    
    manager = BackendManager(config=config)
    
    # Register mock backend
    mock_backend = MockBackend("test_backend")
    manager.register_backend("test_backend", mock_backend)
    
    await manager.initialize_backends()
    
    # Test complete workflow
    result = await manager.generate_with_fallback("Integration test")
    assert result is not None
    
    # Check health
    health = await manager.check_all_backends()
    assert "test_backend" in health
    
    # Get stats
    stats = manager.get_statistics()
    assert "test_backend" in stats


if __name__ == "__main__":
    pytest.main([__file__])
