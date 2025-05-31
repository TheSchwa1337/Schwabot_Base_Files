import os
import json
import time
import logging
from datetime import datetime

R1_DIR = os.path.dirname(os.path.abspath(__file__))
SCHWABOT_HASH_LOG = os.path.join(R1_DIR, '../../schwabot/hashes/BTC_tick_hashes.log')
PATTERNS_FILE = os.path.join(R1_DIR, '../memory_map/profitable_patterns.json')
NEXT_COMMAND = os.path.join(R1_DIR, 'instructions/next_command.json')
LOG_FILE = os.path.join(R1_DIR, 'logs/r1_activity.log')

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

POLL_INTERVAL = 2  # seconds

def load_patterns():
    if os.path.exists(PATTERNS_FILE):
        with open(PATTERNS_FILE, 'r') as f:
            return json.load(f)
    return {}

def main():
    print('[R1InstructionLoop] Monitoring hash log...')
    last_line = ''
    while True:
        if os.path.exists(SCHWABOT_HASH_LOG):
            with open(SCHWABOT_HASH_LOG, 'r') as f:
                lines = f.readlines()
            if lines:
                new_line = lines[-1].strip()
                if new_line != last_line:
                    last_line = new_line
                    patterns = load_patterns()
                    # Check for pattern match (simple substring match for now)
                    for pattern, command in patterns.items():
                        if pattern in new_line:
                            # Write next_command.json
                            with open(NEXT_COMMAND, 'w') as outf:
                                json.dump(command, outf, indent=2)
                            logging.info(f'Pattern matched: {pattern}. Command written: {command}')
                            break
        time.sleep(POLL_INTERVAL)

if __name__ == '__main__':
    main() 