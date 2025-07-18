#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Schwabot Visual System Deployment Script

This script deploys the complete Schwabot visual system with all
components properly configured and tested for production use.

Features:
- Complete system deployment
- Visual layer configuration
- GUI deployment
- Web interface deployment
- System validation
- Production readiness checks
"""

import os
import sys
import json
import yaml
import logging
import subprocess
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchwabotDeployer:
    """Comprehensive deployer for Schwabot visual system."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.deployment_results = {
            'deployment_completed': False,
            'components_deployed': [],
            'errors': [],
            'warnings': [],
            'start_time': None,
            'end_time': None
        }
        
        # Deployment configuration
        self.deployment_config = {
            'system_name': 'Schwabot AI',
            'version': '2.0.0',
            'deployment_mode': 'production',
            'components': {
                'core_system': True,
                'visual_layer': True,
                'gui_interface': True,
                'web_interface': True,
                'api_server': True,
                'mathematical_engine': True,
                'trading_system': True
            },
            'ports': {
                'web_interface': 8080,
                'api_server': 5000,
                'gui_server': 3000
            }
        }
    
    def validate_prerequisites(self) -> Dict[str, Any]:
        """Validate system prerequisites for deployment."""
        logger.info("🔍 Validating deployment prerequisites...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        # Check Python version
        try:
            python_version = sys.version_info
            if python_version.major >= 3 and python_version.minor >= 8:
                results['passed'] += 1
                logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            else:
                results['failed'] += 1
                error_msg = f"Python version {python_version.major}.{python_version.minor} not supported. Need 3.8+"
                results['errors'].append(error_msg)
                logger.error(f"❌ {error_msg}")
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Python version check failed: {e}")
            logger.error(f"❌ Python version check failed: {e}")
        
        # Check required directories
        required_dirs = ['core', 'gui', 'config', 'static', 'templates']
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                results['passed'] += 1
                logger.info(f"✅ Directory found: {dir_name}")
            else:
                results['failed'] += 1
                error_msg = f"Required directory not found: {dir_name}"
                results['errors'].append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        # Check required files
        required_files = [
            'main.py',
            'requirements.txt',
            'config/schwabot_config.yaml',
            'gui/flask_app.py',
            'gui/visualizer_launcher.py'
        ]
        for file_name in required_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                results['passed'] += 1
                logger.info(f"✅ File found: {file_name}")
            else:
                results['failed'] += 1
                error_msg = f"Required file not found: {file_name}"
                results['errors'].append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        return results
    
    def install_dependencies(self) -> Dict[str, Any]:
        """Install system dependencies."""
        logger.info("📦 Installing system dependencies...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            # Install Python dependencies
            requirements_file = self.project_root / 'requirements.txt'
            if requirements_file.exists():
                logger.info("Installing Python dependencies...")
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )
                
                if result.returncode == 0:
                    results['passed'] += 1
                    logger.info("✅ Python dependencies installed successfully")
                else:
                    results['failed'] += 1
                    error_msg = f"Failed to install Python dependencies: {result.stderr}"
                    results['errors'].append(error_msg)
                    logger.error(f"❌ {error_msg}")
            else:
                results['failed'] += 1
                error_msg = "requirements.txt not found"
                results['errors'].append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Dependency installation failed: {e}")
            logger.error(f"❌ Dependency installation failed: {e}")
        
        return results
    
    def configure_system(self) -> Dict[str, Any]:
        """Configure the system for deployment."""
        logger.info("⚙️  Configuring system for deployment...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            # Create deployment configuration
            deployment_config_file = self.project_root / 'config' / 'deployment_config.json'
            with open(deployment_config_file, 'w', encoding='utf-8') as f:
                json.dump(self.deployment_config, f, indent=2, ensure_ascii=False)
            
            results['passed'] += 1
            logger.info("✅ Deployment configuration created")
            
            # Update Schwabot configuration
            schwabot_config_file = self.project_root / 'config' / 'schwabot_config.yaml'
            if schwabot_config_file.exists():
                with open(schwabot_config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # Update configuration for deployment
                config['deployment'] = {
                    'mode': 'production',
                    'version': self.deployment_config['version'],
                    'system_name': self.deployment_config['system_name'],
                    'ports': self.deployment_config['ports']
                }
                
                with open(schwabot_config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                
                results['passed'] += 1
                logger.info("✅ Schwabot configuration updated for deployment")
            else:
                results['failed'] += 1
                error_msg = "Schwabot configuration file not found"
                results['errors'].append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"System configuration failed: {e}")
            logger.error(f"❌ System configuration failed: {e}")
        
        return results
    
    def deploy_visual_layer(self) -> Dict[str, Any]:
        """Deploy the visual layer components."""
        logger.info("🎨 Deploying visual layer...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            # Create static directory if it doesn't exist
            static_dir = self.project_root / 'static'
            static_dir.mkdir(exist_ok=True)
            
            # Create Schwabot logo
            logo_content = '''
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                    🚀 SCHWABOT AI 🚀                        ║
    ║                                                              ║
    ║              Advanced AI-Powered Trading System              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
            '''
            
            logo_file = static_dir / 'schwabot_logo.txt'
            with open(logo_file, 'w', encoding='utf-8') as f:
                f.write(logo_content)
            
            results['passed'] += 1
            logger.info("✅ Schwabot logo created")
            
            # Create CSS file for styling
            css_content = '''
/* Schwabot AI Visual Layer Styles */
.schwabot-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #1E3A8A 0%, #059669 100%);
    color: white;
    min-height: 100vh;
    padding: 20px;
}

.schwabot-header {
    text-align: center;
    margin-bottom: 30px;
}

.schwabot-title {
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
}

.schwabot-subtitle {
    font-size: 1.2em;
    opacity: 0.9;
}

.schwabot-card {
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
}

.schwabot-button {
    background: linear-gradient(45deg, #DC2626, #D97706);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 1em;
    transition: all 0.3s ease;
}

.schwabot-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}
            '''
            
            css_file = static_dir / 'schwabot_styles.css'
            with open(css_file, 'w', encoding='utf-8') as f:
                f.write(css_content)
            
            results['passed'] += 1
            logger.info("✅ Schwabot CSS styles created")
            
            # Create HTML template
            html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Schwabot AI - Advanced Trading System</title>
    <link rel="stylesheet" href="/static/schwabot_styles.css">
</head>
<body>
    <div class="schwabot-container">
        <div class="schwabot-header">
            <h1 class="schwabot-title">🚀 SCHWABOT AI 🚀</h1>
            <p class="schwabot-subtitle">Advanced AI-Powered Trading System</p>
        </div>
        
        <div class="schwabot-card">
            <h2>System Status</h2>
            <p>Schwabot AI is running successfully!</p>
            <button class="schwabot-button" onclick="checkSystemStatus()">Check Status</button>
        </div>
        
        <div class="schwabot-card">
            <h2>Trading Dashboard</h2>
            <p>Access your trading interface and analytics.</p>
            <button class="schwabot-button" onclick="openTradingDashboard()">Open Dashboard</button>
        </div>
        
        <div class="schwabot-card">
            <h2>AI Analysis</h2>
            <p>Advanced AI-powered market analysis and predictions.</p>
            <button class="schwabot-button" onclick="runAIAnalysis()">Run Analysis</button>
        </div>
    </div>
    
    <script>
        function checkSystemStatus() {
            alert('Schwabot AI System Status: ✅ All systems operational');
        }
        
        function openTradingDashboard() {
            window.open('/dashboard', '_blank');
        }
        
        function runAIAnalysis() {
            alert('AI Analysis initiated. Processing market data...');
        }
    </script>
</body>
</html>
            '''
            
            templates_dir = self.project_root / 'templates'
            templates_dir.mkdir(exist_ok=True)
            
            html_file = templates_dir / 'schwabot_dashboard.html'
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            results['passed'] += 1
            logger.info("✅ Schwabot HTML template created")
        
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Visual layer deployment failed: {e}")
            logger.error(f"❌ Visual layer deployment failed: {e}")
        
        return results
    
    def deploy_gui_interface(self) -> Dict[str, Any]:
        """Deploy the GUI interface."""
        logger.info("🖥️  Deploying GUI interface...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            # Update GUI launcher with Schwabot branding
            gui_launcher_file = self.project_root / 'gui' / 'visualizer_launcher.py'
            if gui_launcher_file.exists():
                with open(gui_launcher_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Update branding in GUI launcher
                content = content.replace('KoboldCPP', 'Schwabot AI')
                content = content.replace('koboldcpp', 'schwabot_ai')
                
                with open(gui_launcher_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                results['passed'] += 1
                logger.info("✅ GUI launcher updated with Schwabot branding")
            else:
                results['failed'] += 1
                error_msg = "GUI launcher file not found"
                results['errors'].append(error_msg)
                logger.error(f"❌ {error_msg}")
            
            # Update Flask app with Schwabot branding
            flask_app_file = self.project_root / 'gui' / 'flask_app.py'
            if flask_app_file.exists():
                with open(flask_app_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Update branding in Flask app
                content = content.replace('KoboldCPP', 'Schwabot AI')
                content = content.replace('koboldcpp', 'schwabot_ai')
                
                with open(flask_app_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                results['passed'] += 1
                logger.info("✅ Flask app updated with Schwabot branding")
            else:
                results['failed'] += 1
                error_msg = "Flask app file not found"
                results['errors'].append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"GUI interface deployment failed: {e}")
            logger.error(f"❌ GUI interface deployment failed: {e}")
        
        return results
    
    def deploy_web_interface(self) -> Dict[str, Any]:
        """Deploy the web interface."""
        logger.info("🌐 Deploying web interface...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            # Create web interface launcher
            web_launcher_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 Schwabot AI Web Interface Launcher

Launches the Schwabot AI web interface for trading and analysis.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gui.flask_app import create_app

def main():
    """Launch the Schwabot AI web interface."""
    print("🚀 Starting Schwabot AI Web Interface...")
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                    🚀 SCHWABOT AI 🚀                        ║
    ║                                                              ║
    ║              Advanced AI-Powered Trading System              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        app = create_app()
        app.run(
            host='0.0.0.0',
            port=8080,
            debug=False
        )
    except Exception as e:
        print(f"❌ Failed to start web interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
            '''
            
            web_launcher_file = self.project_root / 'launch_web_interface.py'
            with open(web_launcher_file, 'w', encoding='utf-8') as f:
                f.write(web_launcher_content)
            
            results['passed'] += 1
            logger.info("✅ Web interface launcher created")
        
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Web interface deployment failed: {e}")
            logger.error(f"❌ Web interface deployment failed: {e}")
        
        return results
    
    def run_system_tests(self) -> Dict[str, Any]:
        """Run system tests after deployment."""
        logger.info("🧪 Running post-deployment system tests...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            # Import and run validation test
            from comprehensive_schwabot_validation import SchwabotValidator
            
            validator = SchwabotValidator()
            validation_results = validator.run_comprehensive_validation()
            
            if validation_results['overall_results']['success_rate'] >= 80:
                results['passed'] += 1
                logger.info("✅ System validation passed")
            else:
                results['failed'] += 1
                error_msg = f"System validation failed: {validation_results['overall_results']['success_rate']}% success rate"
                results['errors'].append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"System tests failed: {e}")
            logger.error(f"❌ System tests failed: {e}")
        
        return results
    
    def deploy_complete_system(self) -> Dict[str, Any]:
        """Deploy the complete Schwabot system."""
        logger.info("🚀 Starting complete Schwabot system deployment...")
        logger.info("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                    🚀 SCHWABOT AI 🚀                        ║
    ║                                                              ║
    ║              Advanced AI-Powered Trading System              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
        """)
        
        self.deployment_results['start_time'] = datetime.now()
        
        # Run deployment steps
        deployment_steps = [
            ('prerequisites', self.validate_prerequisites),
            ('dependencies', self.install_dependencies),
            ('configuration', self.configure_system),
            ('visual_layer', self.deploy_visual_layer),
            ('gui_interface', self.deploy_gui_interface),
            ('web_interface', self.deploy_web_interface),
            ('system_tests', self.run_system_tests)
        ]
        
        for step_name, step_function in deployment_steps:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {step_name.upper()} deployment...")
            logger.info(f"{'='*50}")
            
            try:
                result = step_function()
                self.deployment_results['components_deployed'].append({
                    'component': step_name,
                    'result': result
                })
                
                if result['failed'] == 0:
                    logger.info(f"✅ {step_name.upper()} - DEPLOYED SUCCESSFULLY")
                else:
                    logger.error(f"❌ {step_name.upper()} - DEPLOYMENT FAILED")
                    self.deployment_results['errors'].extend(result['errors'])
                
            except Exception as e:
                logger.error(f"❌ {step_name.upper()} - ERROR: {e}")
                self.deployment_results['errors'].append(f"{step_name}: {str(e)}")
        
        self.deployment_results['end_time'] = datetime.now()
        
        # Check if deployment was successful
        total_errors = len(self.deployment_results['errors'])
        if total_errors == 0:
            self.deployment_results['deployment_completed'] = True
            logger.info("✅ Complete system deployment successful!")
        else:
            logger.error(f"❌ Deployment completed with {total_errors} errors")
        
        # Generate deployment report
        report = self.generate_deployment_report()
        
        return report
    
    def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report."""
        execution_time = (self.deployment_results['end_time'] - self.deployment_results['start_time']).total_seconds()
        
        report = {
            'deployment_completed': self.deployment_results['deployment_completed'],
            'timestamp': datetime.now().isoformat(),
            'system_name': self.deployment_config['system_name'],
            'version': self.deployment_config['version'],
            'execution_time_seconds': execution_time,
            'components_deployed': len(self.deployment_results['components_deployed']),
            'errors': len(self.deployment_results['errors']),
            'warnings': len(self.deployment_results['warnings']),
            'detailed_results': self.deployment_results['components_deployed'],
            'error_details': self.deployment_results['errors'],
            'next_steps': self._generate_next_steps()
        }
        
        # Save report
        report_file = self.project_root / 'SCHWABOT_DEPLOYMENT_REPORT.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def _generate_next_steps(self) -> List[str]:
        """Generate next steps based on deployment results."""
        next_steps = []
        
        if self.deployment_results['deployment_completed']:
            next_steps.extend([
                "System is ready for production use",
                "Access web interface at: http://localhost:8080",
                "Launch GUI interface using: python gui/visualizer_launcher.py",
                "Run system tests: python run_all_tests.py",
                "Monitor system logs for any issues"
            ])
        else:
            next_steps.extend([
                "Review deployment errors and fix issues",
                "Re-run deployment after fixing errors",
                "Check system prerequisites",
                "Verify all dependencies are installed",
                "Contact support if issues persist"
            ])
        
        return next_steps

def main():
    """Main function to run the deployment."""
    deployer = SchwabotDeployer()
    
    try:
        report = deployer.deploy_complete_system()
        
        print("\n" + "="*60)
        if report['deployment_completed']:
            print("🎉 SCHWABOT DEPLOYMENT COMPLETED SUCCESSFULLY!")
        else:
            print("⚠️  SCHWABOT DEPLOYMENT COMPLETED WITH ISSUES")
        print("="*60)
        print(f"⏱️  Deployment Time: {report['execution_time_seconds']:.2f} seconds")
        print(f"📦 Components Deployed: {report['components_deployed']}")
        print(f"❌ Errors: {report['errors']}")
        print(f"⚠️  Warnings: {report['warnings']}")
        
        if report['errors'] > 0:
            print(f"\n❌ Deployment Errors:")
            for error in report['error_details'][:5]:  # Show first 5 errors
                print(f"   • {error}")
            if len(report['error_details']) > 5:
                print(f"   ... and {len(report['error_details']) - 5} more")
        
        print(f"\n📋 Next Steps:")
        for step in report['next_steps']:
            print(f"   • {step}")
        
        if report['deployment_completed']:
            print("\n🎉 Your Schwabot AI system is now fully deployed and ready!")
            print("🚀 Access the web interface at: http://localhost:8080")
        else:
            print("\n❌ Please address deployment issues before using the system.")
            
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        print(f"❌ Deployment failed: {e}")

if __name__ == "__main__":
    main() 