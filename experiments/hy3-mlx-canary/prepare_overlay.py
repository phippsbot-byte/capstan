#!/usr/bin/env python3
from pathlib import Path
import json
import os
import shutil

SRC = Path('/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX')
OVERLAY = Path('/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-served')
CUSTOM_MODEL = Path('/Users/nb/LLM/hy3-mlx-canary/hy_v3_mlx.py')

required = ['config.json', 'tokenizer_config.json', 'chat_template.jinja', 'model.safetensors.index.json']
missing = [name for name in required if not (SRC / name).exists()]
shards = sorted(SRC.glob('model-*.safetensors'))
if missing or len(shards) != 34:
    raise SystemExit(f'Source incomplete: missing={missing} shards={len(shards)}/34 at {SRC}')

if OVERLAY.exists() or OVERLAY.is_symlink():
    if OVERLAY.is_symlink() or OVERLAY.is_file():
        OVERLAY.unlink()
    else:
        shutil.rmtree(OVERLAY)
OVERLAY.mkdir(parents=True)

for item in SRC.iterdir():
    if item.name in {'config.json', 'tokenizer_config.json'}:
        continue
    target = OVERLAY / item.name
    os.symlink(item, target, target_is_directory=item.is_dir())

shutil.copy2(CUSTOM_MODEL, OVERLAY / 'hy_v3_mlx.py')

cfg = json.loads((SRC / 'config.json').read_text())
cfg['model_file'] = 'hy_v3_mlx.py'
# mlx_lm's loader handles `quantization` first. Keep the converted MLX quantization map there.
if 'quantization' not in cfg and 'quantization_config' in cfg:
    cfg['quantization'] = cfg['quantization_config']
if 'rope_theta' not in cfg and isinstance(cfg.get('rope_parameters'), dict):
    cfg['rope_theta'] = cfg['rope_parameters'].get('rope_theta', cfg.get('rope_theta', 10000.0))
if 'moe_intermediate_size' not in cfg and 'expert_hidden_dim' in cfg:
    cfg['moe_intermediate_size'] = cfg['expert_hidden_dim']
cfg.setdefault('tie_word_embeddings', False)
(OVERLAY / 'config.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + '\n')

tok = json.loads((SRC / 'tokenizer_config.json').read_text())
# Stock mlx_lm 0.31.3 has no mlx_lm.tool_parsers.hy_v3, so remove this for no-tool canaries.
# Tool-call validation belongs in the Capstan/Hy3 parser workstream, not this flat-MLX smoke.
tok.pop('tool_parser_type', None)
(OVERLAY / 'tokenizer_config.json').write_text(json.dumps(tok, indent=2, ensure_ascii=False) + '\n')

print('overlay', OVERLAY)
print('source', SRC)
print('shards', len(shards))
print('model_file', cfg['model_file'])
print('model_type', cfg.get('model_type'))
print('tokenizer_tool_parser_removed', 'tool_parser_type' not in tok)
print('quantization', {k: cfg.get('quantization', {}).get(k) for k in ['bits','group_size','mode']})
