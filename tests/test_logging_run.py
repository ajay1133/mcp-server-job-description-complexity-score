#!/usr/bin/env python3
import os, json, glob, time
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_server.server import extract_technologies_

# Project root is parent of this tests directory
root_dir = os.path.dirname(os.path.dirname(__file__))
log_dir = os.path.join(root_dir, 'logs', 'responses_traces')
if os.path.isdir(log_dir):
    for f in glob.glob(os.path.join(log_dir, '*.json')):
        try:
            os.remove(f)
        except Exception:
            pass
else:
    os.makedirs(log_dir, exist_ok=True)

res = extract_technologies_("Senior engineer: 4 years React, Node.js, PostgreSQL", "")
print('RESULT_TECHS', list(res.get('technologies', {}).keys()))

time.sleep(0.2)
files = sorted(glob.glob(os.path.join(log_dir, '*.json')))
print('LOG_FILES', [os.path.basename(f) for f in files][-3:])

if files:
    with open(files[-1], 'r', encoding='utf-8') as f:
        payload = json.load(f)
    print('PAYLOAD_KEYS', list(payload.keys()))
    print('TRACE_COUNT', len(payload.get('traces', [])))
    if payload.get('traces'):
        t0 = payload['traces'][0]
        print('TRACE_FUNCTION', t0.get('function'))
        print('TRACE_KEYS', sorted(t0.keys()))
        print('HAS_CPU_DELTAS', 'cpu_user_delta' in t0 and 'cpu_system_delta' in t0)
        print('HAS_MEM_FIELDS', 'memory_before_mb' in t0 and 'memory_after_mb' in t0)
