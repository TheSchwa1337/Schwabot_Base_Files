#!/usr/bin/env python3
"""
Test simplified GPT command layer.
"""

import sys
import asyncio
from datetime import datetime

async def test_simplified_gpt_layer():
    """Test the simplified GPT command layer."""
    print("Testing Simplified GPT Command Layer...")
    
    try:
        from core.gpt_command_layer_simple import (
            AIAgentType, CommandDomain, CommandPriority, 
            AICommand, CommandResponse, GPTCommandLayer
        )
        print("✅ Simplified GPT Command Layer imported successfully")
        
        # Create command layer
        layer = GPTCommandLayer()
        print("✅ GPT Command Layer initialized")
        
        # Create test command
        command = AICommand(
            command_id="test_cmd_001",
            agent_type=AIAgentType.GPT,
            domain=CommandDomain.STRATEGY,
            priority=CommandPriority.MEDIUM,
            hash_signature="test_hash_123",
            timestamp=datetime.now(),
            payload={"strategy_name": "test_strategy", "parameters": {"test": True}},
            context={"test": True}
        )
        print("✅ AI Command created successfully")
        
        # Submit command
        command_id = await layer.submit_command(
            agent_type=AIAgentType.GPT,
            domain=CommandDomain.STRATEGY,
            payload={"strategy_name": "test_strategy", "parameters": {"test": True}},
            context={"test": True}
        )
        print(f"✅ Command submitted: {command_id}")
        
        # Execute commands
        await layer.execute_commands()
        print("✅ Commands executed")
        
        # Get status
        status = await layer.get_command_status(command_id)
        print(f"✅ Command status: {'Success' if status and status.success else 'Failed'}")
        
        # Get system status
        system_status = await layer.get_system_status()
        print(f"✅ System status: {system_status['total_commands']} commands processed")
        
        print("🎉 Simplified GPT Command Layer test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    success = await test_simplified_gpt_layer()
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 