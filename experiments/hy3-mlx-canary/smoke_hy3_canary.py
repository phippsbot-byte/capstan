#!/usr/bin/env python3
import json
import sys
import time
import urllib.request
import urllib.error

BASE = 'http://127.0.0.1:8133/v1'
MODEL = '/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-served'


def post(path, payload, timeout=180):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(BASE + path, data=data, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode('utf-8', 'replace')
    return time.time() - t0, json.loads(body)


def get(path, timeout=10):
    t0 = time.time()
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        body = r.read().decode('utf-8', 'replace')
    return time.time() - t0, json.loads(body)


def main():
    out = {'base': BASE, 'model': MODEL, 'checks': []}
    dt, models = get('/models')
    out['models_latency_sec'] = round(dt, 3)
    out['models'] = models

    probes = [
        ('pong', [{'role': 'user', 'content': 'Reply with exactly the word pong.'}], 'pong', 16),
        ('json', [{'role': 'user', 'content': 'Return exactly this JSON and nothing else: {"ok":true}'}], '{"ok":true}', 32),
        ('math', [{'role': 'user', 'content': 'Return only the number: 84 * 3 / 2'}], '126', 32),
    ]
    for name, messages, expect, max_tokens in probes:
        payload = {
            'model': MODEL,
            'messages': messages,
            'temperature': 0,
            'max_tokens': max_tokens,
            'chat_template_kwargs': {'reasoning_effort': 'no_think'},
        }
        try:
            dt, resp = post('/chat/completions', payload)
            msg = resp['choices'][0]['message']
            content = msg.get('content') or ''
            check = {
                'name': name,
                'latency_sec': round(dt, 3),
                'expect': expect,
                'content': content,
                'finish_reason': resp['choices'][0].get('finish_reason'),
                'usage': resp.get('usage'),
                'pass': content.strip() == expect,
            }
        except Exception as e:
            check = {'name': name, 'error': repr(e), 'pass': False}
        out['checks'].append(check)
        print(json.dumps(check, indent=2), flush=True)

    out['pass'] = all(c.get('pass') for c in out['checks'])
    print('\nSUMMARY')
    print(json.dumps(out, indent=2))
    return 0 if out['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
