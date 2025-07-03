import os
import sys
import subprocess
import traceback

def run_flake8_on_file(filepath):
    try:
        # Ensure the file is readable and not empty
        if not os.path.exists(filepath):
            print(f"File does not exist: {filepath}")
            return 0
        
        if os.path.getsize(filepath) == 0:
            print(f"Skipping empty file: {filepath}")
            return 0
        
        # Try to read the file content to check for encoding issues
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"Encoding issue in file: {filepath}")
            return 0
        
        # Run flake8 with comprehensive checks
        result = subprocess.run(
            ['flake8', filepath, 
             '--max-line-length=100', 
             '--show-source', 
             '--select=E,F,W', 
             '--ignore=E501'], 
            capture_output=True, 
            text=True
        )
        
        # Print results for debugging
        if result.returncode != 0:
            print(f"\n{'='*50}")
            print(f"Issues in {filepath}:")
            print(f"{'='*50}")
            print(result.stdout)
            print(result.stderr)
            print(f"{'='*50}\n")
        
        return result.returncode
    
    except Exception as e:
        print(f"Unexpected error processing {filepath}:")
        print(traceback.format_exc())
        return 1

def main(directories):
    # Ensure directories exist
    valid_directories = [d for d in directories if os.path.isdir(d)]
    if not valid_directories:
        print(f"No valid directories found: {directories}")
        sys.exit(1)
    
    total_errors = 0
    processed_files = 0
    
    for directory in valid_directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    file_errors = run_flake8_on_file(filepath)
                    total_errors += file_errors
                    processed_files += 1
    
    print(f"\nProcessed {processed_files} Python files")
    print(f"Total files with issues: {total_errors}")
    
    sys.exit(1 if total_errors > 0 else 0)

if __name__ == "__main__":
    # Add more directories if needed
    main(["core", "schwabot", "utils", "config"]) 