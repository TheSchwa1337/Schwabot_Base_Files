#!/usr/bin/env python3
"""
Schwabot Deployment Readiness Check

This script performs a comprehensive check to ensure the Schwabot system is ready for deployment
across Windows, macOS, and Linux platforms. It validates:

1. Code quality and formatting
2. Dependencies and imports
3. Cross-platform compatibility
4. Performance requirements
5. API connections
6. Trading functionality
7. Mathematical operations
8. CLI interface readiness
"""

import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class DeploymentReadinessChecker:
    """Comprehensive deployment readiness checker for Schwabot."""

    def __init__(self):
        self.project_root = project_root
        self.results = {}
            "code_quality": {},
            "dependencies": {},
            "cross_platform": {},
            "performance": {},
            "api_connections": {},
            "trading_functionality": {},
            "mathematical_operations": {},
            "cli_interface": {},
            "overall_status": "UNKNOWN"
        }
        self.errors = []
        self.warnings = []

    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run all deployment readiness checks."""
        print("🚀 Starting Schwabot Deployment Readiness Check")
        print("=" * 60)

        # 1. Code Quality Checks
        print("\n📝 Checking Code Quality...")
        self.check_code_quality()

        # 2. Dependencies Check
        print("\n📦 Checking Dependencies...")
        self.check_dependencies()

        # 3. Cross-Platform Compatibility
        print("\n🖥️ Checking Cross-Platform Compatibility...")
        self.check_cross_platform_compatibility()

        # 4. Performance Requirements
        print("\n⚡ Checking Performance Requirements...")
        self.check_performance_requirements()

        # 5. API Connections
        print("\n🌐 Checking API Connections...")
        self.check_api_connections()

        # 6. Trading Functionality
        print("\n📈 Checking Trading Functionality...")
        self.check_trading_functionality()

        # 7. Mathematical Operations
        print("\n🧮 Checking Mathematical Operations...")
        self.check_mathematical_operations()

        # 8. CLI Interface
        print("\n💻 Checking CLI Interface...")
        self.check_cli_interface()

        # Overall Assessment
        self.assess_overall_status()

        return self.results

    def check_code_quality(self):
        """Check code quality using various tools."""
        code_quality = {}
            "flake8": self.run_flake8(),
            "formatting": self.check_formatting(),
            "imports": self.check_imports(),
            "docstrings": self.check_docstrings()
        }
        self.results["code_quality"] = code_quality

    def run_flake8(self) -> Dict[str, Any]:
        """Run flake8 linting."""
        try:
            result = subprocess.run([)]
                sys.executable, "-m", "flake8", "core/", 
                "--max-line-length=120", 
                "--ignore=E203,W503,E501",
                "--count", "--statistics"
            ], capture_output=True, text=True, cwd=self.project_root)

            return {}
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "issues_count": 0 if result.returncode == 0 else "unknown"
            }
        except Exception as e:
            return {}
                "success": False,
                "error": str(e),
                "issues_count": "unknown"
            }

    def check_formatting(self) -> Dict[str, Any]:
        """Check code formatting."""
        formatting_tools = {}
            "black": self.check_black_formatting(),
            "isort": self.check_isort_formatting(),
            "autopep8": self.check_autopep8_available()
        }
        return formatting_tools

    def check_black_formatting(self) -> Dict[str, Any]:
        """Check Black formatting."""
        try:
            # Check if Black is available
            result = subprocess.run([)]
                sys.executable, "-c", "import black; print(black.__version__)"
            ], capture_output=True, text=True)

            if result.returncode != 0:
                return {"available": False, "error": "Black not installed"}

            # Run Black check
            check_result = subprocess.run([)]
                sys.executable, "-m", "black", "--check", "--diff", "core/"
            ], capture_output=True, text=True, cwd=self.project_root)

            return {}
                "available": True,
                "version": result.stdout.strip(),
                "formatted": check_result.returncode == 0,
                "diff": check_result.stdout if check_result.returncode != 0 else ""
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def check_isort_formatting(self) -> Dict[str, Any]:
        """Check isort import formatting."""
        try:
            # Check if isort is available
            result = subprocess.run([)]
                sys.executable, "-c", "import isort; print(isort.__version__)"
            ], capture_output=True, text=True)

            if result.returncode != 0:
                return {"available": False, "error": "isort not installed"}

            # Run isort check
            check_result = subprocess.run([)]
                sys.executable, "-m", "isort", "--check-only", "--diff", "core/"
            ], capture_output=True, text=True, cwd=self.project_root)

            return {}
                "available": True,
                "version": result.stdout.strip(),
                "formatted": check_result.returncode == 0,
                "diff": check_result.stdout if check_result.returncode != 0 else ""
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def check_autopep8_available(self) -> Dict[str, Any]:
        """Check if autopep8 is available."""
        try:
            result = subprocess.run([)]
                sys.executable, "-c", "import autopep8; print(autopep8.__version__)"
            ], capture_output=True, text=True)

            return {}
                "available": result.returncode == 0,
                "version": result.stdout.strip() if result.returncode == 0 else None,
                "error": result.stderr if result.returncode != 0 else None
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def check_imports(self) -> Dict[str, Any]:
        """Check for import issues."""
        core_modules = []
            "core.acceleration_enhancement",
            "core.advanced_tensor_algebra",
            "core.strategy.enhanced_math_ops",
            "core.zpe_core",
            "core.zbe_core",
            "core.clean_trading_pipeline",
            "core.api.integration_manager",
            "utils.cuda_helper"
        ]

        import_results = {}
        for module in core_modules:
            try:
                importlib.import_module(module)
                import_results[module] = {"success": True}
            except ImportError as e:
                import_results[module] = {"success": False, "error": str(e)}
            except Exception as e:
                import_results[module] = {"success": False, "error": f"Unexpected error: {e}"}

        successful_imports = sum(1 for result in import_results.values() if result["success"])
        total_imports = len(import_results)

        return {}
            "results": import_results,
            "success_rate": successful_imports / total_imports,
            "successful": successful_imports,
            "total": total_imports
        }

    def check_docstrings(self) -> Dict[str, Any]:
        """Check docstring coverage."""
        try:
            # Simple docstring check
            core_files = list(Path("core").rglob("*.py"))
            total_files = len(core_files)
            files_with_docstrings = 0

            for file_path in core_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if '"""' in content or "'''" in content: '
                            files_with_docstrings += 1
                except Exception:
                    continue

            return {}
                "total_files": total_files,
                "files_with_docstrings": files_with_docstrings,
                "coverage": files_with_docstrings / total_files if total_files > 0 else 0
            }
        except Exception as e:
            return {"error": str(e)}

    def check_dependencies(self):
        """Check dependencies and requirements."""
        dependencies = {}
            "requirements_files": self.check_requirements_files(),
            "core_dependencies": self.check_core_dependencies(),
            "optional_dependencies": self.check_optional_dependencies(),
            "version_compatibility": self.check_version_compatibility()
        }
        self.results["dependencies"] = dependencies

    def check_requirements_files(self) -> Dict[str, Any]:
        """Check requirements files."""
        req_files = []
            "requirements.txt",
            "requirements_unified.txt",
            "requirements-dev.txt",
            "pyproject.toml"
        ]

        files_status = {}
        for req_file in req_files:
            file_path = self.project_root / req_file
            files_status[req_file] = {}
                "exists": file_path.exists(),
                "size": file_path.stat().st_size if file_path.exists() else 0
            }

        return files_status

    def check_core_dependencies(self) -> Dict[str, Any]:
        """Check core dependencies."""
        core_deps = []
            "numpy", "scipy", "pandas", "matplotlib", "flask",
            "aiohttp", "requests", "pyyaml", "psutil", "cryptography",
            "ccxt", "fastapi", "uvicorn"
        ]

        dep_status = {}
        for dep in core_deps:
            try:
                module = importlib.import_module(dep)
                version = getattr(module, '__version__', 'unknown')
                dep_status[dep] = {"available": True, "version": version}
            except ImportError:
                dep_status[dep] = {"available": False, "error": "Not installed"}
            except Exception as e:
                dep_status[dep] = {"available": False, "error": str(e)}

        available_count = sum(1 for status in dep_status.values() if status["available"])
        return {}
            "dependencies": dep_status,
            "available": available_count,
            "total": len(core_deps),
            "coverage": available_count / len(core_deps)
        }

    def check_optional_dependencies(self) -> Dict[str, Any]:
        """Check optional dependencies."""
        optional_deps = []
            "cupy", "torch", "tensorflow", "plotly", "bokeh",
            "dash", "streamlit", "ta", "numba"
        ]

        dep_status = {}
        for dep in optional_deps:
            try:
                module = importlib.import_module(dep)
                version = getattr(module, '__version__', 'unknown')
                dep_status[dep] = {"available": True, "version": version}
            except ImportError:
                dep_status[dep] = {"available": False, "optional": True}
            except Exception as e:
                dep_status[dep] = {"available": False, "error": str(e)}

        return {"dependencies": dep_status}

    def check_version_compatibility(self) -> Dict[str, Any]:
        """Check Python version compatibility."""
        python_version = sys.version_info

        return {}
            "python_version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            "compatible": python_version >= (3, 8),
            "recommended": python_version >= (3, 10),
            "platform": platform.platform(),
            "architecture": platform.architecture()[0]
        }

    def check_cross_platform_compatibility(self):
        """Check cross-platform compatibility."""
        compatibility = {}
            "current_platform": self.get_platform_info(),
            "path_handling": self.check_path_handling(),
            "file_permissions": self.check_file_permissions(),
            "environment_variables": self.check_environment_variables()
        }
        self.results["cross_platform"] = compatibility

    def get_platform_info(self) -> Dict[str, Any]:
        """Get current platform information."""
        return {}
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation()
        }

    def check_path_handling(self) -> Dict[str, Any]:
        """Check path handling compatibility."""
        try:
            # Test path operations
            test_path = Path("core") / "advanced_tensor_algebra.py"
            return {}
                "pathlib_compatible": test_path.exists(),
                "os_path_compatible": os.path.exists(str(test_path)),
                "separator": os.sep,
                "pathsep": os.pathsep
            }
        except Exception as e:
            return {"error": str(e)}

    def check_file_permissions(self) -> Dict[str, Any]:
        """Check file permissions."""
        try:
            test_file = Path("core") / "__init__.py"
            return {}
                "readable": os.access(test_file, os.R_OK),
                "writable": os.access(test_file, os.W_OK),
                "executable": os.access(test_file, os.X_OK)
            }
        except Exception as e:
            return {"error": str(e)}

    def check_environment_variables(self) -> Dict[str, Any]:
        """Check environment variables."""
        important_vars = ["PATH", "PYTHONPATH", "HOME", "USER"]
        if platform.system() == "Windows":
            important_vars.extend(["USERPROFILE", "APPDATA"])

        env_status = {}
        for var in important_vars:
            env_status[var] = {}
                "set": var in os.environ,
                "value": os.environ.get(var, "Not set")[:100] + "..." if len(os.environ.get(var, "")) > 100 else os.environ.get(var, "Not set")
            }

        return {"variables": env_status}

    def check_performance_requirements(self):
        """Check performance requirements."""
        performance = {}
            "cpu_info": self.get_cpu_info(),
            "memory_info": self.get_memory_info(),
            "gpu_info": self.get_gpu_info(),
            "disk_space": self.get_disk_space()
        }
        self.results["performance"] = performance

    def get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        try:
            import psutil
            return {}
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "cpu_percent": psutil.cpu_percent(interval=1),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
        except Exception as e:
            return {"error": str(e)}

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {}
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
                "free": memory.free
            }
        except Exception as e:
            return {"error": str(e)}

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        gpu_info = {"cuda_available": False, "cupy_available": False}

        try:
            import cupy as cp
            gpu_info["cupy_available"] = True
            gpu_info["cupy_version"] = cp.__version__
            gpu_info["cuda_version"] = cp.cuda.runtime.runtimeGetVersion()
            gpu_info["device_count"] = cp.cuda.runtime.getDeviceCount()
        except Exception as e:
            gpu_info["cupy_error"] = str(e)

        try:
            from utils.cuda_helper import get_cuda_status
            cuda_status = get_cuda_status()
            gpu_info["cuda_helper_status"] = cuda_status
        except Exception as e:
            gpu_info["cuda_helper_error"] = str(e)

        return gpu_info

    def get_disk_space(self) -> Dict[str, Any]:
        """Get disk space information."""
        try:
            import psutil
            disk_usage = psutil.disk_usage('.')
            return {}
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "percent": (disk_usage.used / disk_usage.total) * 100
            }
        except Exception as e:
            return {"error": str(e)}

    def check_api_connections(self):
        """Check API connections."""
        api_status = {}
            "exchange_apis": self.check_exchange_apis(),
            "market_data": self.check_market_data_apis(),
            "network_connectivity": self.check_network_connectivity()
        }
        self.results["api_connections"] = api_status

    def check_exchange_apis(self) -> Dict[str, Any]:
        """Check exchange API availability."""
        try:
            import ccxt
            exchanges = ['binance', 'coinbase', 'kraken', 'bitfinex']
            exchange_status = {}

            for exchange_name in exchanges:
                try:
                    exchange_class = getattr(ccxt, exchange_name)
                    exchange = exchange_class({'sandbox': True})
                    exchange_status[exchange_name] = {}
                        "available": True,
                        "sandbox": exchange.sandbox,
                        "has_public_api": hasattr(exchange, 'fetch_ticker')
                    }
                except Exception as e:
                    exchange_status[exchange_name] = {}
                        "available": False,
                        "error": str(e)
                    }

            return {}
                "ccxt_version": ccxt.__version__,
                "exchanges": exchange_status
            }
        except Exception as e:
            return {"error": str(e)}

    def check_market_data_apis(self) -> Dict[str, Any]:
        """Check market data API availability."""
        try:
            from core.api.integration_manager import IntegrationManager
            api_manager = IntegrationManager()
            return {}
                "integration_manager": True,
                "status": "Available"
            }
        except Exception as e:
            return {}
                "integration_manager": False,
                "error": str(e)
            }

    def check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity."""
        try:
            import requests
            test_urls = []
                "https://api.binance.com/api/v3/ping",
                "https://api.coinbase.com/v2/time",
                "https://httpbin.org/get"
            ]

            connectivity = {}
            for url in test_urls:
                try:
                    response = requests.get(url, timeout=5)
                    connectivity[url] = {}
                        "reachable": response.status_code == 200,
                        "status_code": response.status_code,
                        "response_time": response.elapsed.total_seconds()
                    }
                except Exception as e:
                    connectivity[url] = {}
                        "reachable": False,
                        "error": str(e)
                    }

            return connectivity
        except Exception as e:
            return {"error": str(e)}

    def check_trading_functionality(self):
        """Check trading functionality."""
        trading = {}
            "core_systems": self.check_core_trading_systems(),
            "mathematical_engine": self.check_mathematical_engine(),
            "strategy_execution": self.check_strategy_execution()
        }
        self.results["trading_functionality"] = trading

    def check_core_trading_systems(self) -> Dict[str, Any]:
        """Check core trading systems."""
        systems = {}
            "zpe_core": self.test_import("core.zpe_core", "ZPECore"),
            "zbe_core": self.test_import("core.zbe_core", "ZBECore"),
            "dual_state_router": self.test_import("core.system.dual_state_router", "DualStateRouter"),
            "clean_trading_pipeline": self.test_import("core.clean_trading_pipeline", "CleanTradingPipeline")
        }

        available_systems = sum(1 for system in systems.values() if system["available"])
        return {}
            "systems": systems,
            "available": available_systems,
            "total": len(systems),
            "coverage": available_systems / len(systems)
        }

    def check_mathematical_engine(self) -> Dict[str, Any]:
        """Check mathematical engine."""
        try:
            from core.advanced_tensor_algebra import AdvancedTensorAlgebra
            from core.strategy.enhanced_math_ops import get_enhancement_status

            # Test basic operations
            tensor_algebra = AdvancedTensorAlgebra()
            enhancement_status = get_enhancement_status()

            return {}
                "tensor_algebra": True,
                "enhancement_available": enhancement_status["enhancement_available"],
                "cuda_available": enhancement_status["cuda_available"],
                "operations": enhancement_status["operations"]
            }
        except Exception as e:
            return {"error": str(e)}

    def check_strategy_execution(self) -> Dict[str, Any]:
        """Check strategy execution capabilities."""
        try:
            from core.acceleration_enhancement import get_acceleration_enhancement

            enhancement = get_acceleration_enhancement()
            report = enhancement.get_enhancement_report()

            return {}
                "acceleration_enhancement": True,
                "cuda_available": report["cuda_available"],
                "system_integration": report["existing_system_integration"],
                "status": report["status"]
            }
        except Exception as e:
            return {"error": str(e)}

    def check_mathematical_operations(self):
        """Check mathematical operations."""
        math_ops = {}
            "tensor_operations": self.test_tensor_operations(),
            "profit_vectors": self.test_profit_vectors(),
            "entropy_calculations": self.test_entropy_calculations()
        }
        self.results["mathematical_operations"] = math_ops

    def test_tensor_operations(self) -> Dict[str, Any]:
        """Test tensor operations."""
        try:
            from core.advanced_tensor_algebra import AdvancedTensorAlgebra
            import numpy as np

            tensor_algebra = AdvancedTensorAlgebra()

            # Test basic tensor operations
            A = np.random.rand(10, 10)
            B = np.random.rand(10, 10)

            start_time = time.time()
            result = tensor_algebra.tensor_dot_fusion(A, B)
            execution_time = time.time() - start_time

            return {}
                "success": True,
                "execution_time": execution_time,
                "result_shape": result.shape,
                "result_sum": float(np.sum(result))
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_profit_vectors(self) -> Dict[str, Any]:
        """Test profit vector operations."""
        try:
            from core.strategy.enhanced_math_ops import enhanced_profit_vectorization
            import numpy as np

            profits = np.random.rand(100) * 100
            weights = np.random.rand(100)
            weights = weights / np.sum(weights)

            start_time = time.time()
            result = enhanced_profit_vectorization()
                profits, weights, 
                entropy=0.7, profit_weight=0.8, 
                use_enhancement=True
            )
            execution_time = time.time() - start_time

            return {}
                "success": True,
                "execution_time": execution_time,
                "total_profit": float(np.sum(result)),
                "efficiency": float(np.sum(result) / np.sum(profits))
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_entropy_calculations(self) -> Dict[str, Any]:
        """Test entropy calculations."""
        try:
            from core.acceleration_enhancement import get_acceleration_enhancement

            enhancement = get_acceleration_enhancement()

            # Test ZPE enhancement
            zpe_data = enhancement.calculate_zpe_enhancement()
                tick_delta=0.5, registry_swing=0.3
            )

            # Test ZBE enhancement
            zbe_data = enhancement.calculate_zbe_enhancement()
                failure_count=2, recent_weight=0.7
            )

            # Test combined entropy
            combined_entropy = enhancement.get_combined_entropy_score(zpe_data, zbe_data)

            return {}
                "success": True,
                "zpe_entropy": zpe_data.entropy_score,
                "zbe_entropy": zbe_data.entropy_score,
                "combined_entropy": combined_entropy
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def check_cli_interface(self):
        """Check CLI interface."""
        cli_status = {}
            "launcher_scripts": self.check_launcher_scripts(),
            "command_interface": self.check_command_interface(),
            "configuration": self.check_configuration()
        }
        self.results["cli_interface"] = cli_status

    def check_launcher_scripts(self) -> Dict[str, Any]:
        """Check launcher scripts."""
        scripts = []
            "schwabot_enhanced_launcher.py",
            "test_schwabot_integration.py",
            "deployment_readiness_check.py"
        ]

        script_status = {}
        for script in scripts:
            script_path = self.project_root / script
            script_status[script] = {}
                "exists": script_path.exists(),
                "executable": os.access(script_path, os.X_OK) if script_path.exists() else False,
                "size": script_path.stat().st_size if script_path.exists() else 0
            }

        return script_status

    def check_command_interface(self) -> Dict[str, Any]:
        """Check command interface."""
        try:
            # Check if main entry points are available
            entry_points = {}
                "main_launcher": self.test_import("schwabot_enhanced_launcher", "main"),
                "integration_test": self.test_import("test_schwabot_integration", "main"),
                "tensor_test": self.test_import("test_tensor_profit_system", "main")
            }

            available_entry_points = sum(1 for ep in entry_points.values() if ep["available"])

            return {}
                "entry_points": entry_points,
                "available": available_entry_points,
                "total": len(entry_points),
                "coverage": available_entry_points / len(entry_points)
            }
        except Exception as e:
            return {"error": str(e)}

    def check_configuration(self) -> Dict[str, Any]:
        """Check configuration files."""
        config_files = []
            "pyproject.toml",
            ".flake8",
            "mypy.ini",
            "setup.cfg"
        ]

        config_status = {}
        for config_file in config_files:
            config_path = self.project_root / config_file
            config_status[config_file] = {}
                "exists": config_path.exists(),
                "size": config_path.stat().st_size if config_path.exists() else 0
            }

        return config_status

    def test_import(self, module_name: str, class_name: Optional[str] = None) -> Dict[str, Any]:
        """Test import of a module and optionally a class."""
        try:
            module = importlib.import_module(module_name)
            if class_name:
                cls = getattr(module, class_name)
                return {"available": True, "module": module_name, "class": class_name}
            return {"available": True, "module": module_name}
        except Exception as e:
            return {"available": False, "module": module_name, "error": str(e)}

    def assess_overall_status(self):
        """Assess overall deployment readiness status."""
        scores = []

        # Code quality score
        code_quality = self.results["code_quality"]
        if code_quality.get("flake8", {}).get("success", False):
            scores.append(1.0)
        else:
            scores.append(0.5)

        # Dependencies score
        dependencies = self.results["dependencies"]
        dep_coverage = dependencies.get("core_dependencies", {}).get("coverage", 0)
        scores.append(dep_coverage)

        # Cross-platform score
        cross_platform = self.results["cross_platform"]
        if cross_platform.get("current_platform", {}).get("system"):
            scores.append(1.0)
        else:
            scores.append(0.5)

        # Trading functionality score
        trading = self.results["trading_functionality"]
        trading_coverage = trading.get("core_systems", {}).get("coverage", 0)
        scores.append(trading_coverage)

        # Mathematical operations score
        math_ops = self.results["mathematical_operations"]
        math_success = sum(1 for op in math_ops.values() if op.get("success", False))
        math_score = math_success / len(math_ops) if math_ops else 0
        scores.append(math_score)

        # CLI interface score
        cli = self.results["cli_interface"]
        cli_coverage = cli.get("command_interface", {}).get("coverage", 0)
        scores.append(cli_coverage)

        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else 0

        if overall_score >= 0.9:
            status = "READY_FOR_DEPLOYMENT"
        elif overall_score >= 0.7:
            status = "MOSTLY_READY"
        elif overall_score >= 0.5:
            status = "NEEDS_IMPROVEMENT"
        else:
            status = "NOT_READY"

        self.results["overall_status"] = status
        self.results["overall_score"] = overall_score

        # Generate recommendations
        self.generate_recommendations()

    def generate_recommendations(self):
        """Generate recommendations for deployment readiness."""
        recommendations = []

        # Code quality recommendations
        code_quality = self.results["code_quality"]
        if not code_quality.get("formatting", {}).get("black", {}).get("available", False):
            recommendations.append("Install Black formatter: pip install black")

        if not code_quality.get("formatting", {}).get("isort", {}).get("available", False):
            recommendations.append("Install isort: pip install isort")

        # Dependencies recommendations
        dependencies = self.results["dependencies"]
        missing_deps = []
        for dep, status in dependencies.get("core_dependencies", {}).get("dependencies", {}).items():
            if not status.get("available", False):
                missing_deps.append(dep)

        if missing_deps:
            recommendations.append(f"Install missing dependencies: pip install {' '.join(missing_deps)}")

        # Performance recommendations
        performance = self.results["performance"]
        if not performance.get("gpu_info", {}).get("cupy_available", False):
            recommendations.append("Consider installing CuPy for GPU acceleration: pip install cupy")

        # Configuration recommendations
        if not (self.project_root / ".flake8").exists():
            recommendations.append("Create .flake8 configuration file")

        if not (self.project_root / "mypy.ini").exists():
            recommendations.append("Create mypy.ini configuration file")

        self.results["recommendations"] = recommendations

    def print_summary(self):
        """Print deployment readiness summary."""
        print("\n" + "=" * 80)
        print("📋 DEPLOYMENT READINESS SUMMARY")
        print("=" * 80)

        status = self.results["overall_status"]
        score = self.results.get("overall_score", 0)

        status_emoji = {}
            "READY_FOR_DEPLOYMENT": "🎉",
            "MOSTLY_READY": "✅",
            "NEEDS_IMPROVEMENT": "⚠️",
            "NOT_READY": "❌"
        }

        print(f"\n{status_emoji.get(status, '❓')} Overall Status: {status}")
        print(f"📊 Overall Score: {score:.1%}")

        # Print section summaries
        sections = []
            ("Code Quality", "code_quality"),
            ("Dependencies", "dependencies"),
            ("Cross-Platform", "cross_platform"),
            ("Performance", "performance"),
            ("API Connections", "api_connections"),
            ("Trading Functionality", "trading_functionality"),
            ("Mathematical Operations", "mathematical_operations"),
            ("CLI Interface", "cli_interface")
        ]

        print("\n📊 Section Details:")
        for section_name, section_key in sections:
            section_data = self.results.get(section_key, {})
            if section_data:
                print(f"  {section_name}: {'✅' if section_data else '❌'}")

        # Print recommendations
        recommendations = self.results.get("recommendations", [])
        if recommendations:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

        print("\n🚀 Next Steps:")
        if status == "READY_FOR_DEPLOYMENT":
            print("  • Your system is ready for deployment!")
            print("  • Consider running final integration tests")
            print("  • Deploy to your target environment")
        elif status == "MOSTLY_READY":
            print("  • Address the recommendations above")
            print("  • Run this check again after fixes")
            print("  • Consider additional testing")
        else:
            print("  • Address critical issues first")
            print("  • Focus on missing dependencies")
            print("  • Ensure core functionality works")

        print("\n📱 Platform Support:")
        platform_info = self.results.get("cross_platform", {}).get("current_platform", {})
        current_system = platform_info.get("system", "Unknown")
        print(f"  • Current: {current_system}")
        print(f"  • Windows: {'✅' if current_system == 'Windows' else '🔄'}")
        print(f"  • macOS: {'✅' if current_system == 'Darwin' else '🔄'}")
        print(f"  • Linux: {'✅' if current_system == 'Linux' else '🔄'}")


def main():
    """Main function to run deployment readiness check."""
    checker = DeploymentReadinessChecker()
    results = checker.run_comprehensive_check()

    # Print detailed summary
    checker.print_summary()

    # Save results to file
    results_file = Path("deployment_readiness_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to: {results_file}")

    return results


if __name__ == "__main__":
    main() 