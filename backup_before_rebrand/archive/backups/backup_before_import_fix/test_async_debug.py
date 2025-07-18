#!/usr/bin/env python3
"""
Simple async test script to debug CLI async issues.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

async def test_async():
    print(" Async function started")
    await asyncio.sleep(0.1)
    print(" Async function completed")
    return 0

async def main():
    print(" Main async function started")
    result = await test_async()
    print(f" Result: {result}")
    return result

if __name__ == "__main__":
    print(" Starting async test...")
    result = asyncio.run(main())
    print(f" Final result: {result}")
    sys.exit(result) 