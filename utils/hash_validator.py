import sys
import json
import argparse

def validate_hash(hash_value, pattern_file):
    with open(pattern_file, 'r') as f:
        patterns = json.load(f)
    for pattern in patterns:
        if pattern in hash_value:
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description='Hash Validator')
    parser.add_argument('--hash', required=True, help='Hash value to check')
    parser.add_argument('--patterns', required=True, help='Pattern file (JSON)')
    args = parser.parse_args()
    result = validate_hash(args.hash, args.patterns)
    print('MATCH' if result else 'NO MATCH')

if __name__ == '__main__':
    main() 