#!/usr/bin/env python3
"""
APEX DIRECTOR System Test
Validates core functionality of the system
"""

import sys
import asyncio
from pathlib import Path

# Add the package to path for imports
sys.path.insert(0, str(Path(__file__).parent))

async def test_basic_functionality():
    """Test basic system functionality"""
    print("🧪 Testing APEX DIRECTOR Core System...")
    print("=" * 50)
    
    try:
        # Test 1: Import all components
        print("1. Testing imports...")
        from apex_director.core import (
            get_orchestrator,
            get_backend_manager,
            get_asset_manager,
            get_estimator,
            get_checkpoint_manager,
            get_config
        )
        print("   ✅ All imports successful")
        
        # Test 2: Configuration
        print("\\n2. Testing configuration...")
        config = get_config()
        enabled_backends = config.get_enabled_backends()
        print(f"   ✅ Enabled backends: {enabled_backends}")
        
        # Test 3: Backend Manager
        print("\\n3. Testing backend manager...")
        backend_manager = get_backend_manager()
        backend_statuses = backend_manager.get_all_backend_status()
        print(f"   ✅ Initialized {len(backend_statuses)} backends")
        
        # Test 4: Asset Manager  
        print("\\n4. Testing asset manager...")
        asset_manager = get_asset_manager()
        storage_stats = asset_manager.get_storage_stats()
        print(f"   ✅ Asset manager initialized: {storage_stats['total_files']} files")
        
        # Test 5: Estimator
        print("\\n5. Testing estimator...")
        estimator = get_estimator()
        test_estimate = estimator.estimate_generation(
            {"width": 512, "height": 512, "steps": 20, "quality_level": 3},
            "A simple test image"
        )
        print(f"   ✅ Estimation working: ${test_estimate.estimated_cost:.4f}")
        
        # Test 6: Checkpoint Manager
        print("\\n6. Testing checkpoint manager...")
        checkpoint_manager = get_checkpoint_manager()
        checkpoint_status = checkpoint_manager.get_checkpoint_status()
        print(f"   ✅ Checkpoint system ready: {checkpoint_status['total_checkpoints']} checkpoints")
        
        # Test 7: Orchestrator
        print("\\n7. Testing orchestrator...")
        orchestrator = get_orchestrator()
        system_status = orchestrator.get_system_status()
        print(f"   ✅ Orchestrator ready: {system_status['orchestrator']['status']}")
        
        print("\\n" + "=" * 50)
        print("✅ All core functionality tests passed!")
        print("\\n🎉 APEX DIRECTOR core system is ready for use!")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_job_submission():
    """Test job submission and processing"""
    print("\\n🔄 Testing job submission...")
    
    try:
        from apex_director import start_orchestrator, submit_generation_job, stop_orchestrator
        
        # Start orchestrator
        await start_orchestrator()
        
        # Submit test job
        job_id = await submit_generation_job(
            prompt="A test image of a cat",
            priority=3,
            generation_params={
                "width": 512,
                "height": 512,
                "steps": 10
            }
        )
        
        print(f"   ✅ Job submitted: {job_id}")
        
        # Wait a bit for processing
        await asyncio.sleep(2)
        
        # Get status
        from apex_director import get_orchestrator
        orchestrator = get_orchestrator()
        status = orchestrator.get_queue_status()
        
        print(f"   ✅ Queue status: {status}")
        
        # Stop orchestrator
        await stop_orchestrator()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Job submission test failed: {e}")
        return False

def cleanup_metadata_files():
    """Clean up metadata files before and after tests."""
    print("\\n🧹 Cleaning up metadata files...")
    metadata_dir = Path("assets/metadata")
    if metadata_dir.exists():
        for f in metadata_dir.glob("*.json"):
            if "backup" not in f.name:
                try:
                    f.unlink()
                    print(f"   🗑️  Removed {f.name}")
                except OSError as e:
                    print(f"   ❌ Error removing {f.name}: {e}")

async def main():
    """Main function to run the system tests."""
    print("APEX DIRECTOR System Validation")
    print("=" * 50)
    
    cleanup_metadata_files()
    
    try:
        # Run basic functionality tests
        basic_success = await test_basic_functionality()
        
        if basic_success:
            # Run job submission test
            job_success = await test_job_submission()

            if job_success:
                print("\\n🎊 All tests completed successfully!")
                print("The APEX DIRECTOR core system is fully functional.")
            else:
                print("\\n⚠️  Basic functionality works, but job processing has issues.")
        else:
            print("\\n❌ Core system tests failed. Please check the implementation.")

    finally:
        cleanup_metadata_files()

if __name__ == "__main__":
    asyncio.run(main())