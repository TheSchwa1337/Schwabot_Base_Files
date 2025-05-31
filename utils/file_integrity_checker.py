import os
import hashlib
import json
import argparse

def compute_file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def check_integrity(reference_file):
    with open(reference_file, 'r') as f:
        ref_hashes = json.load(f)
    for path, ref_hash in ref_hashes.items():
        if not os.path.exists(path):
            print(f'MISSING: {path}')
            continue
        actual_hash = compute_file_hash(path)
        if actual_hash != ref_hash:
            print(f'TAMPERED: {path}')
        else:
            print(f'OK: {path}')

def main():
    parser = argparse.ArgumentParser(description='File Integrity Checker')
    parser.add_argument('--reference', required=True, help='Reference hash file (JSON)')
    args = parser.parse_args()
    check_integrity(args.reference)

if __name__ == '__main__':
    main() 