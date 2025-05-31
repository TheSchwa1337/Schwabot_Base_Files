#!/bin/bash
# Install cron job for btc_tick_hash.py
CRON_JOB="* * * * * /usr/bin/python3 /schwabot/btc_tick_hash.py >> /schwabot/logs/hash_debug.log 2>&1"
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
echo "Cron job installed for btc_tick_hash.py." 