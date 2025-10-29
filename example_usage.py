#!/usr/bin/env python3
"""
APEX DIRECTOR Example Usage
Demonstrates how to use the core system components
"""

import asyncio
import sys
from pathlib import Path

# Add the package to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from apex_director.core import (
    get_orchestrator,
    get_backend_manager, 
    get_asset_manager,
    get_estimator,
    get_checkpoint_manager,
    get_config,
    start_orchestrator,
    stop_orchestrator,
    submit_generation_job,
    get_system_status
)

async def example_basic_usage():
    """Basic usage example"""
    print("=== APEX DIRECTOR Basic Usage Example ===\n")
    
    # 1. Get system components
    orchestrator = get_orchestrator()
    backend_manager = get_backend_manager()
    asset_manager = get_asset_manager()
    estimator = get_estimator()
    
    print("1. System components initialized")
    
    # 2. Check backend status
    backend_statuses = backend_manager.get_all_backend_status()
    print(f"\\n2. Backend Status:")
    for name, status in backend_statuses.items():
        print(f"   {name}: {status.status} (success rate: {status.success_rate:.2%})")
    
    # 3. Submit a single job
    print("\\n3. Submitting generation job...")
    job_id = await submit_generation_job(
        prompt="A majestic lion in a savanna at sunset",
        priority=3,
        generation_params={
            "width": 512,
            "height": 512, 
            "steps": 20,
            "quality_level": 3
        }
    )
    print(f"   Job submitted: {job_id}")
    
    return job_id

async def example_batch_processing():
    """Batch processing example"""
    print("\\n=== Batch Processing Example ===\\n")
    
    # Prepare batch jobs
    jobs = [
        {
            "prompt": "A serene mountain lake with reflection",
            "priority": 4,
            "generation_params": {"width": 768, "height": 768}
        },
        {
            "prompt": "A cyberpunk city at night with neon lights", 
            "priority": 2,
            "generation_params": {"width": 512, "height": 512}
        },
        {
            "prompt": "A cozy cottage in a forest clearing",
            "priority": 3,
            "generation_params": {"width": 640, "height": 480}
        }
    ]
    
    orchestrator = get_orchestrator()
    
    print("1. Submitting batch jobs...")
    job_ids = await orchestrator.submit_batch(jobs)
    print(f"   Batch submitted: {len(job_ids)} jobs")
    
    for i, job_id in enumerate(job_ids):
        print(f"   Job {i+1}: {job_id}")
    
    return job_ids

async def example_estimation():
    """Cost and time estimation example"""
    print("\\n=== Estimation Example ===\\n")
    
    estimator = get_estimator()
    
    # Test different job configurations
    test_params = [
        {
            "name": "Quick test (512x512, low quality)",
            "params": {"width": 512, "height": 512, "steps": 10, "quality_level": 2},
            "prompt": "A simple flower"
        },
        {
            "name": "High quality (1024x1024, more steps)", 
            "params": {"width": 1024, "height": 1024, "steps": 50, "quality_level": 5},
            "prompt": "A highly detailed fantasy landscape with dragons and magic"
        }
    ]
    
    for test in test_params:
        print(f"1. Estimating: {test['name']}")
        estimate = estimator.estimate_generation(test['params'], test['prompt'])
        
        print(f"   Cost: ${estimate.estimated_cost:.4f}")
        print(f"   Time: {estimate.estimated_time_seconds:.1f} seconds")
        print(f"   Confidence: {estimate.confidence_score:.2%}")
        print(f"   Suggested backends: {', '.join(estimate.backend_suggestions[:2])}")
        print()
    
    return test_params

async def example_asset_management():
    """Asset management example"""
    print("\\n=== Asset Management Example ===\\n")
    
    asset_manager = get_asset_manager()
    
    # Create a project
    print("1. Creating project...")
    project_id = asset_manager.create_project(
        name="Demo Project",
        description="Example project for demonstration",
        settings={"auto_cleanup": False}
    )
    print(f"   Project created: {project_id}")
    
    # Get storage statistics
    print("\\n2. Storage statistics:")
    stats = asset_manager.get_storage_stats()
    print(f"   Total files: {stats['total_files']}")
    print(f"   Total size: {stats['total_size_mb']:.2f} MB")
    print(f"   Projects: {stats['projects_count']}")
    
    return project_id

async def example_checkpoints():
    """Checkpoint system example"""
    print("\\n=== Checkpoint System Example ===\\n")
    
    checkpoint_manager = get_checkpoint_manager()
    
    # Create manual checkpoint
    print("1. Creating manual checkpoint...")
    checkpoint_id = await checkpoint_manager.create_checkpoint("demo_checkpoint")
    print(f"   Checkpoint created: {checkpoint_id}")
    
    # List checkpoints
    print("\\n2. Available checkpoints:")
    checkpoints = await checkpoint_manager.list_checkpoints()
    for cp in checkpoints:
        print(f"   {cp['id']}: {cp['timestamp']} ({cp['job_count']} jobs)")
    
    # Get checkpoint status
    print("\\n3. Checkpoint status:")
    status = checkpoint_manager.get_checkpoint_status()
    print(f"   Auto checkpointing: {status['auto_checkpoint_enabled']}")
    print(f"   Total checkpoints: {status['total_checkpoints']}")
    print(f"   Disk usage: {status['disk_usage_mb']:.2f} MB")
    
    return checkpoint_id

async def example_system_monitoring():
    """System monitoring example"""
    print("\\n=== System Monitoring Example ===\\n")
    
    # Get comprehensive system status
    status = get_system_status()
    
    print("1. Orchestrator Status:")
    orchestrator = status['orchestrator']
    print(f"   Status: {orchestrator['status']}")
    print(f"   Uptime: {orchestrator['uptime_seconds']/3600:.1f} hours")
    print(f"   Active processors: {orchestrator['active_processors']}")
    
    print("\\n2. Job Queue Status:")
    jobs = status['jobs']
    print(f"   Queued: {jobs['total_queued']}")
    print(f"   Active: {jobs['active_jobs']}")
    print(f"   Completed: {jobs['completed_jobs']}")
    print(f"   Failed: {jobs['failed_jobs']}")
    
    print("\\n3. Performance Metrics:")
    perf = status['performance']
    print(f"   Total processed: {perf['total_jobs_processed']}")
    print(f"   Success rate: {perf['success_rate']:.2%}")
    print(f"   Total cost: ${perf['total_cost_accumulated']:.4f}")
    
    return status

async def main():
    """Main example demonstrating all system capabilities"""
    print("APEX DIRECTOR Core System Demo")
    print("=" * 50)
    
    try:
        # Start the orchestrator
        print("\\nStarting orchestrator...")
        await start_orchestrator()
        
        # Run examples
        job_id = await example_basic_usage()
        job_ids = await example_batch_processing()
        await example_estimation()
        project_id = await example_asset_management()
        checkpoint_id = await example_checkpoints()
        system_status = await example_system_monitoring()
        
        # Wait a bit to let jobs process
        print("\\n" + "=" * 50)
        print("Waiting for jobs to process...")
        await asyncio.sleep(10)
        
        # Final status check
        final_status = get_system_status()
        print("\\nFinal Status:")
        print(f"Completed jobs: {final_status['jobs']['completed_jobs']}")
        print(f"Active jobs: {final_status['jobs']['active_jobs']}")
        
        print("\\n" + "=" * 50)
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"\\nError during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean shutdown
        print("\\nShutting down orchestrator...")
        await stop_orchestrator()
        print("Shutdown complete")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())