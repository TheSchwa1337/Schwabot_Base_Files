#!/usr/bin/env python3
"""
Test single module import.
"""

import sys

def test_gpt_layer():
    """Test GPT command layer import."""
    print("Testing GPT Command Layer import...")
    
    try:
        print("Attempting to import...")
        from core.gpt_command_layer import AIAgentType
        print("✅ AIAgentType imported successfully")
        
        from core.gpt_command_layer import CommandDomain
        print("✅ CommandDomain imported successfully")
        
        from core.gpt_command_layer import CommandPriority
        print("✅ CommandPriority imported successfully")
        
        from core.gpt_command_layer import AICommand
        print("✅ AICommand imported successfully")
        
        from core.gpt_command_layer import CommandResponse
        print("✅ CommandResponse imported successfully")
        
        print("🎉 All GPT Command Layer classes imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gpt_layer()
    sys.exit(0 if success else 1) 