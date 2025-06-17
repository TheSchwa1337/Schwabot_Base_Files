"""
Config Audit CLI Tool
===================

Command-line tool for validating YAML configuration files,
checking paths, and verifying schema compliance.
"""

import argparse
import sys
from pathlib import Path
import yaml
from typing import Dict, List, Tuple

# Expected schema for each config file
EXPECTED_SCHEMAS = {
    'recursive.yaml': {
        'mode': str,
        'psi_threshold': float,
        'max_depth': int
    },
    'vault.yaml': {
        'memory_retention': int,
        'similarity_threshold': float
    },
    'matrix_response_paths.yaml': {
        'fault_responses': dict,
        'default_action': str
    },
    'braid.yaml': {
        'patterns': list,
        'confidence_threshold': float
    },
    'logging.yaml': {
        'level': str,
        'format': str,
        'output': str
    }
}

def validate_yaml_syntax(filepath: Path) -> Tuple[bool, str]:
    """Validate YAML syntax"""
    try:
        with open(filepath, 'r') as f:
            yaml.safe_load(f)
        return True, "OK"
    except Exception as e:
        return False, f"YAML syntax error: {str(e)}"

def validate_schema(filepath: Path, schema: Dict) -> Tuple[bool, List[str]]:
    """Validate YAML against expected schema"""
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
            
        errors = []
        for key, expected_type in schema.items():
            if key not in config:
                errors.append(f"Missing required key: {key}")
            elif not isinstance(config[key], expected_type):
                errors.append(
                    f"Invalid type for {key}: expected {expected_type.__name__}, "
                    f"got {type(config[key]).__name__}"
                )
                
        return len(errors) == 0, errors
    except Exception as e:
        return False, [f"Error loading config: {str(e)}"]

def validate_paths(config_dir: Path) -> Dict[str, Dict[str, any]]:
    """Validate all config files in directory"""
    results = {}
    
    for filename, schema in EXPECTED_SCHEMAS.items():
        filepath = config_dir / filename
        file_results = {
            'exists': filepath.exists(),
            'syntax_ok': False,
            'schema_ok': False,
            'errors': []
        }
        
        if file_results['exists']:
            # Check YAML syntax
            syntax_ok, syntax_error = validate_yaml_syntax(filepath)
            file_results['syntax_ok'] = syntax_ok
            if not syntax_ok:
                file_results['errors'].append(syntax_error)
                
            # Check schema if syntax is valid
            if syntax_ok:
                schema_ok, schema_errors = validate_schema(filepath, schema)
                file_results['schema_ok'] = schema_ok
                file_results['errors'].extend(schema_errors)
        else:
            file_results['errors'].append("File does not exist")
            
        results[filename] = file_results
        
    return results

def print_results(results: Dict[str, Dict[str, any]]):
    """Print validation results"""
    print("\nConfig Validation Results:")
    print("=" * 50)
    
    all_ok = True
    for filename, result in results.items():
        status = "[OK]" if result['exists'] and result['syntax_ok'] and result['schema_ok'] else "[FAIL]"
        print(f"\n{status} {filename}")
        
        if not result['exists']:
            print("  File is missing")
            all_ok = False
            continue
            
        if not result['syntax_ok']:
            print("  YAML syntax is invalid")
            all_ok = False
            
        if not result['schema_ok']:
            print("  Schema validation failed")
            all_ok = False
            
        if result['errors']:
            print("  Errors:")
            for error in result['errors']:
                print(f"    - {error}")
                
    print("\n" + "=" * 50)
    print(f"Overall Status: {'[OK] All OK' if all_ok else '[FAIL] Issues Found'}")
    
def main():
    parser = argparse.ArgumentParser(description="Validate YAML configuration files")
    parser.add_argument(
        '--config-dir',
        type=str,
        default='config',
        help='Path to config directory (default: config)'
    )
    
    args = parser.parse_args()
    config_dir = Path(args.config_dir)
    
    if not config_dir.exists():
        print(f"Error: Config directory '{config_dir}' does not exist")
        sys.exit(1)
        
    results = validate_paths(config_dir)
    print_results(results)
    
    # Exit with error if any issues found
    if not all(
        r['exists'] and r['syntax_ok'] and r['schema_ok']
        for r in results.values()
    ):
        sys.exit(1)

if __name__ == '__main__':
    main() 