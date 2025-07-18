#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Command Sequencer for Schwabot Trading System
===============================================

Sequences and manages AI commands for the trading system.
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AICommand:
    """AI command structure."""
    command_id: str
    command_type: str
    parameters: Dict[str, Any]
    priority: int
    created_at: float
    executed_at: Optional[float]

class AICommandSequencer:
    """Sequences AI commands for execution."""
    
    def __init__(self):
        """Initialize the AI command sequencer."""
        self.command_queue: List[AICommand] = []
        self.executed_commands: Dict[str, AICommand] = {}
        self.command_handlers: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_handler(self, command_type: str, handler: Callable) -> None:
        """Register a handler for a command type."""
        self.command_handlers[command_type] = handler
        self.logger.info(f"Registered handler for command type: {command_type}")
    
    def add_command(self, command_type: str, parameters: Dict[str, Any], 
                   priority: int = 1) -> str:
        """Add a command to the queue."""
        command_id = f"{command_type}_{int(time.time() * 1000)}"
        
        command = AICommand(
            command_id=command_id,
            command_type=command_type,
            parameters=parameters,
            priority=priority,
            created_at=time.time(),
            executed_at=None
        )
        
        self.command_queue.append(command)
        self.command_queue.sort(key=lambda x: x.priority, reverse=True)
        
        self.logger.info(f"Added command to queue: {command_id}")
        return command_id
    
    async def execute_commands(self) -> None:
        """Execute commands in the queue."""
        while self.command_queue:
            command = self.command_queue.pop(0)
            
            if command.command_type in self.command_handlers:
                try:
                    handler = self.command_handlers[command.command_type]
                    await handler(command.parameters)
                    command.executed_at = time.time()
                    self.executed_commands[command.command_id] = command
                    self.logger.info(f"Executed command: {command.command_id}")
                except Exception as e:
                    self.logger.error(f"Failed to execute command {command.command_id}: {e}")
            else:
                self.logger.warning(f"No handler for command type: {command.command_type}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get the status of the command queue."""
        return {
            "queue_length": len(self.command_queue),
            "executed_count": len(self.executed_commands),
            "registered_handlers": list(self.command_handlers.keys())
        }

# Global instance
ai_command_sequencer = AICommandSequencer()

def get_ai_command_sequencer() -> AICommandSequencer:
    """Get the global AI command sequencer instance."""
    return ai_command_sequencer
