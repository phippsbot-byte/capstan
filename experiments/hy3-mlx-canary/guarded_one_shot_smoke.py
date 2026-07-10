#!/usr/bin/env python3
import json
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

MANIFEST = '/Users/nb/LLM/hy3-mlx-canary/hy3-preview-mlx-canary.toml'
BASE = 'http://127.0.0.1:8133/v1'
MODEL = '/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-served'
OUT = Path('/Volumes/ModelSSD/logs/hy3-mlx-canary/guarded-one-shot-result.json')
MAX_SWAP_GIB = 90.0
TIMEOUT_SEC = 300

result = {
    'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'max_swap_gib': MAX_SWAP_GIB,
    'samples': [],
    'model': MODEL,
}
stop_event = threading.Event()
kill_reason = None


def run(cmd, timeout=None):
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return {'cmd': cmd, 'returncode': p.returncode, 'output': p.stdout[-4000:]}


def swap_gib():
    out = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True)
    m = re.search(r'used = ([0-9.]+)([MG])', out)
    if not m:
        return None, out.strip()
    val = float(m.group(1))
    if m.group(2) == 'M':
        val /= 1024.0
    return val, out.strip()


def stop_model():
    try:
        subprocess.run(['modelctl', '-m', MANIFEST, 'stop'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    except Exception:
        pass


def monitor():
    global kill_reason
    while not stop_event.is_set():
        val, raw = swap_gib()
        sample = {'t': round(time.time() - t0, 1), 'swap_gib': val, 'raw': raw}
        result['samples'].append(sample)
        if val is not None and val > MAX_SWAP_GIB:
            kill_reason = f'swap {val:.2f} GiB exceeded {MAX_SWAP_GIB:.2f} GiB'
            result['kill_reason'] = kill_reason
            stop_model()
            return
        time.sleep(2)


t0 = time.time()
try:
    result['pre_swap'] = swap_gib()[0]
    result['stop_before'] = run(['modelctl', '-m', MANIFEST, 'stop'], timeout=60)
    result['start'] = run(['modelctl', '-m', MANIFEST, 'start'], timeout=120)
    result['wait'] = run(['modelctl', '-m', MANIFEST, 'wait', '--timeout', '900'], timeout=900)
    if result['wait']['returncode'] != 0:
        raise RuntimeError('server did not become ready')

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()

    payload = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': 'Reply with exactly the word pong.'}],
        'temperature': 0,
        'max_tokens': 8,
        'chat_template_kwargs': {'reasoning_effort': 'no_think'},
    }
    req = urllib.request.Request(BASE + '/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    req_t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as r:
            body = r.read().decode('utf-8', 'replace')
        result['request_latency_sec'] = round(time.time() - req_t0, 3)
        result['response_raw'] = body[:8000]
        parsed = json.loads(body)
        result['response'] = parsed
        content = parsed['choices'][0]['message'].get('content') or ''
        result['content'] = content
        result['pass'] = content.strip() == 'pong'
    except Exception as e:
        result['request_latency_sec'] = round(time.time() - req_t0, 3)
        result['request_error'] = repr(e)
        result['pass'] = False
finally:
    stop_event.set()
    result['post_stop'] = run(['modelctl', '-m', MANIFEST, 'stop'], timeout=60)
    result['post_swap'] = swap_gib()[0]
    result['ended_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    result['elapsed_sec'] = round(time.time() - t0, 3)
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result.get('pass') else 1)
