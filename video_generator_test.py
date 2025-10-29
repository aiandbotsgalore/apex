#!/usr/bin/env python3
"""
REAL Video Generator using actual AI toolkits
This creates actual MP4 videos using the video generation APIs.
"""

import asyncio
import os
import sys
from pathlib import Path
import json
import time

# Import the toolkit functions
try:
    from batch_text_to_video import batch_text_to_video
    TOOLKIT_AVAILABLE = True
except ImportError:
    TOOLKIT_AVAILABLE = False

async def generate_actual_video(prompt: str, output_path: str, duration: int = 6, resolution: str = "768P") -> bool:
    """Generate an actual video using the AI toolkit"""
    try:
        print(f"🎬 Generating REAL video...")
        print(f"📝 Prompt: {prompt[:100]}...")
        print(f"📁 Output: {output_path}")
        print(f"⏱️  Duration: {duration}s")
        print(f"📺 Resolution: {resolution}")
        
        if not TOOLKIT_AVAILABLE:
            print("❌ Video toolkit not available")
            return False
        
        # Call the actual batch_text_to_video function
        result = await batch_text_to_video(
            count=1,
            prompt_list=[prompt],
            output_file_list=[output_path],
            duration_list=[duration],
            resolution_list=[resolution]
        )
        
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            print(f"✅ SUCCESS: Video generated ({file_size / (1024*1024):.1f} MB)")
            return True
        else:
            print(f"❌ Video file not created")
            return False
            
    except Exception as e:
        print(f"❌ Video generation error: {e}")
        return False

async def generate_multiple_videos():
    """Generate multiple test videos to demonstrate the system"""
    test_prompts = [
        {
            "prompt": "A futuristic city with neon lights and flying cars, cyberpunk style, cinematic lighting, high quality",
            "filename": "cyberpunk_city.mp4",
            "concept": "Cyberpunk Cityscape"
        },
        {
            "prompt": "A dragon flying over a medieval castle, epic fantasy scene, golden hour lighting",
            "filename": "dragon_castle.mp4", 
            "concept": "Dragon Over Castle"
        },
        {
            "prompt": "Abstract geometric patterns dancing to electronic music, colorful, vibrant, modern art style",
            "filename": "abstract_dance.mp4",
            "concept": "Abstract Patterns"
        }
    ]
    
    print("🎬 Generating multiple test videos...")
    print("=" * 50)
    
    for i, test in enumerate(test_prompts, 1):
        print(f"\n📹 Video {i}/3: {test['concept']}")
        print("-" * 30)
        
        success = await generate_actual_video(
            prompt=test["prompt"],
            output_path=f"/workspace/{test['filename']}",
            duration=6,
            resolution="768P"
        )
        
        if success:
            print(f"✅ {test['filename']} - SUCCESS!")
        else:
            print(f"❌ {test['filename']} - FAILED!")
        
        if i < len(test_prompts):
            print("⏳ Waiting 2 seconds before next video...")
            await asyncio.sleep(2)
    
    print("\n" + "=" * 50)
    print("🎉 Video generation test complete!")
    
    # List generated files
    print("\n📁 Generated files:")
    for test in test_prompts:
        file_path = Path(f"/workspace/{test['filename']}")
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"  ✅ {test['filename']} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {test['filename']} (missing)")

if __name__ == "__main__":
    print("🚀 REAL Video Generator Test")
    print("Using actual AI video generation toolkits")
    print("=" * 50)
    
    if TOOLKIT_AVAILABLE:
        asyncio.run(generate_multiple_videos())
    else:
        print("❌ Video generation toolkits not available")
        print("The web interface will run but cannot generate videos")
        
    print("\n✅ Test complete!")