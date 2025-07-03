import asyncio
import argparse
import json
import logging
from typing import Dict, Any

from .integration_hub import create_schwabot_hub, SchwabotIntegrationHub

class SchwabotCLI:
    """
    Command-line interface for Schwabot Integration Hub
    Provides various commands to interact with the system
    """
    
    def __init__(self):
        """Initialize the CLI with argument parsing"""
        self.logger = logging.getLogger('SchwabotCLI')
        self.parser = self._create_argument_parser()
        self.hub: SchwabotIntegrationHub = None
    
    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """
        Create the argument parser with various commands
        
        Commands:
        - init: Initialize the system
        - status: Get system status
        - collect-tensors: Collect tensor data
        - process-market: Process market intelligence
        - execute-strategy: Execute a trading strategy
        """
        parser = argparse.ArgumentParser(description="Schwabot Integration Hub CLI")
        
        # Global options
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--capital', type=float, default=100000.0, help='Initial trading capital')
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Init command
        init_parser = subparsers.add_parser('init', help='Initialize Schwabot systems')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Get system status')
        
        # Collect Tensors command
        tensor_parser = subparsers.add_parser('collect-tensors', help='Collect tensor data from sources')
        
        # Process Market command
        market_parser = subparsers.add_parser('process-market', help='Process market intelligence')
        
        # Execute Strategy command
        strategy_parser = subparsers.add_parser('execute-strategy', help='Execute a trading strategy')
        strategy_parser.add_argument('--params', type=str, help='JSON string of strategy parameters')
        
        return parser
    
    async def _initialize_hub(self):
        """Initialize the Schwabot Integration Hub"""
        args = self.parser.parse_args()
        self.hub = await create_schwabot_hub(
            initial_capital=args.capital, 
            debug=args.debug
        )
    
    def _output_json(self, data: Dict[str, Any]):
        """Output data as formatted JSON"""
        print(json.dumps(data, indent=2))
    
    async def run(self):
        """Run the CLI based on the provided command"""
        args = self.parser.parse_args()
        
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG if args.debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize hub
        await self._initialize_hub()
        
        # Command routing
        if args.command == 'init':
            self.logger.info("Schwabot systems initialized")
            self._output_json({"status": "initialized"})
        
        elif args.command == 'status':
            status = self.hub.get_system_status()
            self._output_json(status)
        
        elif args.command == 'collect-tensors':
            tensors = await self.hub.collect_tensor_data()
            self._output_json({
                "tensor_count": len(tensors),
                "tensors": tensors[:5]  # Show first 5 for brevity
            })
        
        elif args.command == 'process-market':
            market_intel = await self.hub.process_market_intelligence()
            self._output_json(market_intel)
        
        elif args.command == 'execute-strategy':
            if not args.params:
                self.logger.error("Strategy parameters required")
        return

            try:
                strategy_params = json.loads(args.params)
                results = await self.hub.execute_trading_strategy(strategy_params)
                self._output_json(results)
            except json.JSONDecodeError:
                self.logger.error("Invalid JSON for strategy parameters")

        else:
            self.parser.print_help()

def main():
    """Entry point for the Schwabot CLI"""
    cli = SchwabotCLI()
    asyncio.run(cli.run())

if __name__ == '__main__':
        main()
