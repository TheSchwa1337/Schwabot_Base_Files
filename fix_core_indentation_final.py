#!/usr/bin/env python3
"""
Final fix for core/__init__.py indentation issues.
"""

def fix_core_indentation_final():
    """Fix all remaining indentation issues in core/__init__.py."""
    
    file_path = "core/__init__.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix specific indentation issues
    fixed_lines = []
    for i, line in enumerate(lines):
        # Fix line 295: for module_name, description in core_modules:
        if i == 294 and 'for module_name, description in core_modules:' in line:
            fixed_line = '        for module_name, description in core_modules:\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 297: initialization_status["modules"].append(module_result)
        elif i == 296 and 'initialization_status["modules"].append(module_result)' in line:
            fixed_line = '                initialization_status["modules"].append(module_result)\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 299: module_result = {
        elif i == 298 and 'module_result = {' in line and 'name": module_name,' in lines[i+1]:
            fixed_line = '                module_result = {\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 300: "name": module_name,
        elif i == 299 and '"name": module_name,' in line:
            fixed_line = '                    "name": module_name,\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 301: "description": description,
        elif i == 300 and '"description": description,' in line:
            fixed_line = '                    "description": description,\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 302: "status": "error",
        elif i == 301 and '"status": "error",' in line:
            fixed_line = '                    "status": "error",\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 303: "error": str(e),
        elif i == 302 and '"error": str(e),' in line:
            fixed_line = '                    "error": str(e),\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 305: initialization_status["modules"].append(module_result)
        elif i == 304 and 'initialization_status["modules"].append(module_result)' in line:
            fixed_line = '                initialization_status["modules"].append(module_result)\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 306: initialization_status["errors"].append(f"Module {module_name}: {e}")
        elif i == 305 and 'initialization_status["errors"].append(f"Module {module_name}: {e}")' in line:
            fixed_line = '                initialization_status["errors"].append(f"Module {module_name}: {e}")\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 318: for component_name, class_name, description in core_components:
        elif i == 317 and 'for component_name, class_name, description in core_components:' in line:
            fixed_line = '        for component_name, class_name, description in core_components:\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 320: component_result = {
        elif i == 319 and 'component_result = {' in line and 'name": component_name,' in lines[i+1]:
            fixed_line = '                component_result = {\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 321: "name": component_name,
        elif i == 320 and '"name": component_name,' in line:
            fixed_line = '                    "name": component_name,\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 322: "class": class_name,
        elif i == 321 and '"class": class_name,' in line:
            fixed_line = '                    "class": class_name,\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 323: "description": description,
        elif i == 322 and '"description": description,' in line:
            fixed_line = '                    "description": description,\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 324: "status": "success",
        elif i == 323 and '"status": "success",' in line:
            fixed_line = '                    "status": "success",\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 325: "timestamp": datetime.now().isoformat()
        elif i == 324 and '"timestamp": datetime.now().isoformat()' in line:
            fixed_line = '                    "timestamp": datetime.now().isoformat()\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 327: initialization_status["components"].append(component_result)
        elif i == 326 and 'initialization_status["components"].append(component_result)' in line:
            fixed_line = '                initialization_status["components"].append(component_result)\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 329: component_result = {
        elif i == 328 and 'component_result = {' in line and 'name": component_name,' in lines[i+1]:
            fixed_line = '                component_result = {\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 330: "name": component_name,
        elif i == 329 and '"name": component_name,' in line:
            fixed_line = '                    "name": component_name,\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 331: "class": class_name,
        elif i == 330 and '"class": class_name,' in line:
            fixed_line = '                    "class": class_name,\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 332: "description": description,
        elif i == 331 and '"description": description,' in line:
            fixed_line = '                    "description": description,\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 333: "status": "error",
        elif i == 332 and '"status": "error",' in line:
            fixed_line = '                    "status": "error",\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 334: "error": str(e),
        elif i == 333 and '"error": str(e),' in line:
            fixed_line = '                    "error": str(e),\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 336: initialization_status["components"].append(component_result)
        elif i == 335 and 'initialization_status["components"].append(component_result)' in line:
            fixed_line = '                initialization_status["components"].append(component_result)\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 337: initialization_status["errors"].append(f"Component {component_name}: {e}")
        elif i == 336 and 'initialization_status["errors"].append(f"Component {component_name}: {e}")' in line:
            fixed_line = '                initialization_status["errors"].append(f"Component {component_name}: {e}")\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 340: successful_modules = sum(1 for m in initialization_status["modules"] if m["status"] == "success")
        elif i == 339 and 'successful_modules = sum(1 for m in initialization_status["modules"] if m["status"] == "success")' in line:
            fixed_line = '        successful_modules = sum(1 for m in initialization_status["modules"] if m["status"] == "success")\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 341: successful_components = sum(1 for c in initialization_status["components"] if c["status"] == "success")
        elif i == 340 and 'successful_components = sum(1 for c in initialization_status["components"] if c["status"] == "success")' in line:
            fixed_line = '        successful_components = sum(1 for c in initialization_status["components"] if c["status"] == "success")\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 343: if successful_modules == len(core_modules) and successful_components == len(core_components):
        elif i == 342 and 'if successful_modules == len(core_modules) and successful_components == len(core_components):' in line:
            fixed_line = '        if successful_modules == len(core_modules) and successful_components == len(core_components):\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 344: initialization_status["status"] = "success"
        elif i == 343 and 'initialization_status["status"] = "success"' in line:
            fixed_line = '            initialization_status["status"] = "success"\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 345: elif successful_modules > len(core_modules) // 2:
        elif i == 344 and 'elif successful_modules > len(core_modules) // 2:' in line:
            fixed_line = '        elif successful_modules > len(core_modules) // 2:\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 346: initialization_status["status"] = "partial"
        elif i == 345 and 'initialization_status["status"] = "partial"' in line:
            fixed_line = '            initialization_status["status"] = "partial"\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 347: else:
        elif i == 346 and 'else:' in line:
            fixed_line = '        else:\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 348: initialization_status["status"] = "failed"
        elif i == 347 and 'initialization_status["status"] = "failed"' in line:
            fixed_line = '            initialization_status["status"] = "failed"\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 350: initialization_status["summary"] = {
        elif i == 349 and 'initialization_status["summary"] = {' in line:
            fixed_line = '        initialization_status["summary"] = {\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 351: "total_modules": len(core_modules),
        elif i == 350 and '"total_modules": len(core_modules),' in line:
            fixed_line = '            "total_modules": len(core_modules),\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 352: "successful_modules": successful_modules,
        elif i == 351 and '"successful_modules": successful_modules,' in line:
            fixed_line = '            "successful_modules": successful_modules,\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 353: "total_components": len(core_components),
        elif i == 352 and '"total_components": len(core_components),' in line:
            fixed_line = '            "total_components": len(core_components),\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 354: "successful_components": successful_components,
        elif i == 353 and '"successful_components": successful_components,' in line:
            fixed_line = '            "successful_components": successful_components,\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 355: "error_count": len(initialization_status["errors"])
        elif i == 354 and '"error_count": len(initialization_status["errors"])' in line:
            fixed_line = '            "error_count": len(initialization_status["errors"])\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 357: logger.info(f"Core system initialization: {initialization_status['status']}")
        elif i == 356 and 'logger.info(f"Core system initialization: {initialization_status[\'status\']}")' in line:
            fixed_line = '        logger.info(f"Core system initialization: {initialization_status[\'status\']}")\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 358: return initialization_status
        elif i == 357 and 'return initialization_status' in line:
            fixed_line = '        return initialization_status\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 360: except Exception as e:
        elif i == 359 and 'except Exception as e:' in line:
            fixed_line = '    except Exception as e:\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 361: logger.error(f"Core system initialization failed: {e}")
        elif i == 360 and 'logger.error(f"Core system initialization failed: {e}")' in line:
            fixed_line = '        logger.error(f"Core system initialization failed: {e}")\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 362: return {
        elif i == 361 and 'return {' in line:
            fixed_line = '        return {\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 363: "status": "failed",
        elif i == 362 and '"status": "failed",' in line:
            fixed_line = '            "status": "failed",\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 364: "error": str(e),
        elif i == 363 and '"error": str(e),' in line:
            fixed_line = '            "error": str(e),\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 365: "timestamp": datetime.now().isoformat(),
        elif i == 364 and '"timestamp": datetime.now().isoformat(),' in line:
            fixed_line = '            "timestamp": datetime.now().isoformat(),\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 366: "modules": [],
        elif i == 365 and '"modules": [],' in line:
            fixed_line = '            "modules": [],\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 367: "components": [],
        elif i == 366 and '"components": [],' in line:
            fixed_line = '            "components": [],\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 368: "errors": [str(e)]
        elif i == 367 and '"errors": [str(e)]' in line:
            fixed_line = '            "errors": [str(e)]\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 369: }
        elif i == 368 and '}' in line:
            fixed_line = '        }\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 372: def check_system_health() -> Dict[str, Any]:
        elif i == 371 and 'def check_system_health() -> Dict[str, Any]:' in line:
            fixed_line = '\n\ndef check_system_health() -> Dict[str, Any]:\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 374: """Check the overall health of the Schwabot system."""
        elif i == 373 and '"""Check the overall health of the Schwabot system."""' in line:
            fixed_line = '    """Check the overall health of the Schwabot system."""\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 375: try:
        elif i == 374 and 'try:' in line:
            fixed_line = '    try:\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 377: health_status = {
        elif i == 376 and 'health_status = {' in line:
            fixed_line = '        health_status = {\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 378: "timestamp": datetime.now().isoformat(),
        elif i == 377 and '"timestamp": datetime.now().isoformat(),' in line:
            fixed_line = '            "timestamp": datetime.now().isoformat(),\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 379: "overall_health": "unknown",
        elif i == 378 and '"overall_health": "unknown",' in line:
            fixed_line = '            "overall_health": "unknown",\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 380: "components": {},
        elif i == 379 and '"components": {},' in line:
            fixed_line = '            "components": {},\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 381: "warnings": [],
        elif i == 380 and '"warnings": [],' in line:
            fixed_line = '            "warnings": [],\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 382: "errors": []
        elif i == 381 and '"errors": []' in line:
            fixed_line = '            "errors": []\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 383: }
        elif i == 382 and '}' in line:
            fixed_line = '        }\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 385: # Define health check functions
        elif i == 384 and '# Define health check functions' in line:
            fixed_line = '        # Define health check functions\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 386: health_checks = {
        elif i == 385 and 'health_checks = {' in line:
            fixed_line = '        health_checks = {\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 387: "core_modules": lambda: len([m for m in initialize_core_system()["modules"] if m["status"] == "success"]) > 0,
        elif i == 386 and '"core_modules": lambda: len([m for m in initialize_core_system()["modules"] if m["status"] == "success"]) > 0,' in line:
            fixed_line = '            "core_modules": lambda: len([m for m in initialize_core_system()["modules"] if m["status"] == "success"]) > 0,\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 388: "typing_schemas": lambda: True,  # Basic check - if we can import, it's working
        elif i == 387 and '"typing_schemas": lambda: True,  # Basic check - if we can import, it\'s working' in line:
            fixed_line = '            "typing_schemas": lambda: True,  # Basic check - if we can import, it\'s working\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 389: "fault_bus": lambda: True,  # Basic check
        elif i == 388 and '"fault_bus": lambda: True,  # Basic check' in line:
            fixed_line = '            "fault_bus": lambda: True,  # Basic check\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 390: "mathematical_validation": lambda: True,  # Basic check
        elif i == 389 and '"mathematical_validation": lambda: True,  # Basic check' in line:
            fixed_line = '            "mathematical_validation": lambda: True,  # Basic check\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 391: }
        elif i == 390 and '}' in line:
            fixed_line = '        }\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 393: healthy_components = 0
        elif i == 392 and 'healthy_components = 0' in line:
            fixed_line = '        healthy_components = 0\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 394: total_components = len(health_checks)
        elif i == 393 and 'total_components = len(health_checks)' in line:
            fixed_line = '        total_components = len(health_checks)\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 396: for component_name, health_check in health_checks.items():
        elif i == 395 and 'for component_name, health_check in health_checks.items():' in line:
            fixed_line = '        for component_name, health_check in health_checks.items():\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 398: is_healthy = health_check()
        elif i == 397 and 'is_healthy = health_check()' in line:
            fixed_line = '                is_healthy = health_check()\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 399: health_status["components"][component_name] = {
        elif i == 398 and 'health_status["components"][component_name] = {' in line:
            fixed_line = '                health_status["components"][component_name] = {\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 400: "status": "healthy" if is_healthy else "unhealthy",
        elif i == 399 and '"status": "healthy" if is_healthy else "unhealthy",' in line:
            fixed_line = '                    "status": "healthy" if is_healthy else "unhealthy",\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 401: "timestamp": datetime.now().isoformat()
        elif i == 400 and '"timestamp": datetime.now().isoformat()' in line:
            fixed_line = '                    "timestamp": datetime.now().isoformat()\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 402: }
        elif i == 401 and '}' in line:
            fixed_line = '                }\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 403: if is_healthy:
        elif i == 402 and 'if is_healthy:' in line:
            fixed_line = '                if is_healthy:\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 404: healthy_components += 1
        elif i == 403 and 'healthy_components += 1' in line:
            fixed_line = '                    healthy_components += 1\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 405: else:
        elif i == 404 and 'else:' in line:
            fixed_line = '                else:\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 406: health_status["warnings"].append(f"Component {component_name} is unhealthy")
        elif i == 405 and 'health_status["warnings"].append(f"Component {component_name} is unhealthy")' in line:
            fixed_line = '                    health_status["warnings"].append(f"Component {component_name} is unhealthy")\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 408: except Exception as e:
        elif i == 407 and 'except Exception as e:' in line:
            fixed_line = '            except Exception as e:\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 409: health_status["components"][component_name] = {
        elif i == 408 and 'health_status["components"][component_name] = {' in line:
            fixed_line = '                health_status["components"][component_name] = {\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 410: "status": "error",
        elif i == 409 and '"status": "error",' in line:
            fixed_line = '                    "status": "error",\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 411: "error": str(e),
        elif i == 410 and '"error": str(e),' in line:
            fixed_line = '                    "error": str(e),\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 412: "timestamp": datetime.now().isoformat()
        elif i == 411 and '"timestamp": datetime.now().isoformat()' in line:
            fixed_line = '                    "timestamp": datetime.now().isoformat()\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 413: }
        elif i == 412 and '}' in line:
            fixed_line = '                }\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 414: health_status["errors"].append(f"Component {component_name}: {e}")
        elif i == 413 and 'health_status["errors"].append(f"Component {component_name}: {e}")' in line:
            fixed_line = '                health_status["errors"].append(f"Component {component_name}: {e}")\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 416: # Determine overall health
        elif i == 415 and '# Determine overall health' in line:
            fixed_line = '        # Determine overall health\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 417: if healthy_components == total_components:
        elif i == 416 and 'if healthy_components == total_components:' in line:
            fixed_line = '        if healthy_components == total_components:\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 418: health_status["overall_health"] = "healthy"
        elif i == 417 and 'health_status["overall_health"] = "healthy"' in line:
            fixed_line = '            health_status["overall_health"] = "healthy"\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 419: elif healthy_components > total_components // 2:
        elif i == 418 and 'elif healthy_components > total_components // 2:' in line:
            fixed_line = '        elif healthy_components > total_components // 2:\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 420: health_status["overall_health"] = "degraded"
        elif i == 419 and 'health_status["overall_health"] = "degraded"' in line:
            fixed_line = '            health_status["overall_health"] = "degraded"\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 421: else:
        elif i == 420 and 'else:' in line:
            fixed_line = '        else:\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 422: health_status["overall_health"] = "unhealthy"
        elif i == 421 and 'health_status["overall_health"] = "unhealthy"' in line:
            fixed_line = '            health_status["overall_health"] = "unhealthy"\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 424: health_status["summary"] = {
        elif i == 423 and 'health_status["summary"] = {' in line:
            fixed_line = '        health_status["summary"] = {\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 425: "total_components": total_components,
        elif i == 424 and '"total_components": total_components,' in line:
            fixed_line = '            "total_components": total_components,\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 426: "healthy_components": healthy_components,
        elif i == 425 and '"healthy_components": healthy_components,' in line:
            fixed_line = '            "healthy_components": healthy_components,\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 427: "unhealthy_components": total_components - healthy_components,
        elif i == 426 and '"unhealthy_components": total_components - healthy_components,' in line:
            fixed_line = '            "unhealthy_components": total_components - healthy_components,\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 428: "error_count": len(health_status["errors"]),
        elif i == 427 and '"error_count": len(health_status["errors"]),' in line:
            fixed_line = '            "error_count": len(health_status["errors"]),\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 429: "warning_count": len(health_status["warnings"])
        elif i == 428 and '"warning_count": len(health_status["warnings"])' in line:
            fixed_line = '            "warning_count": len(health_status["warnings"])\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 430: }
        elif i == 429 and '}' in line:
            fixed_line = '        }\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 432: logger.info(f"System health check: {health_status['overall_health']} ({healthy_components}/{total_components} components healthy)")
        elif i == 431 and 'logger.info(f"System health check: {health_status[\'overall_health\']} ({healthy_components}/{total_components} components healthy)")' in line:
            fixed_line = '        logger.info(f"System health check: {health_status[\'overall_health\']} ({healthy_components}/{total_components} components healthy)")\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 433: return health_status
        elif i == 432 and 'return health_status' in line:
            fixed_line = '        return health_status\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 435: except Exception as e:
        elif i == 434 and 'except Exception as e:' in line:
            fixed_line = '    except Exception as e:\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 436: logger.error(f"System health check failed: {e}")
        elif i == 435 and 'logger.error(f"System health check failed: {e}")' in line:
            fixed_line = '        logger.error(f"System health check failed: {e}")\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 437: return {
        elif i == 436 and 'return {' in line:
            fixed_line = '        return {\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 438: "overall_health": "error",
        elif i == 437 and '"overall_health": "error",' in line:
            fixed_line = '            "overall_health": "error",\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 439: "error": str(e),
        elif i == 438 and '"error": str(e),' in line:
            fixed_line = '            "error": str(e),\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 440: "timestamp": datetime.now().isoformat(),
        elif i == 439 and '"timestamp": datetime.now().isoformat(),' in line:
            fixed_line = '            "timestamp": datetime.now().isoformat(),\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 441: "components": {},
        elif i == 440 and '"components": {},' in line:
            fixed_line = '            "components": {},\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 442: "warnings": [],
        elif i == 441 and '"warnings": [],' in line:
            fixed_line = '            "warnings": [],\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 443: "errors": [str(e)]
        elif i == 442 and '"errors": [str(e)]' in line:
            fixed_line = '            "errors": [str(e)]\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        # Fix line 444: }
        elif i == 443 and '}' in line:
            fixed_line = '        }\n'
            print(f"  🔧 Fixed indentation at line {i+1}")
        else:
            fixed_line = line
        
        fixed_lines.append(fixed_line)
    
    # Write the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"✅ Fixed final indentation issues in {file_path}")

if __name__ == "__main__":
    fix_core_indentation_final() 