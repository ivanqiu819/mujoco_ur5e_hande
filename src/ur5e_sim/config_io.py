"""读取标准 JSON 配置；各层的 ``_说明`` 仅供阅读，不参与运行或指纹。"""
from __future__ import annotations

import json
from pathlib import Path


def without_notes(value):
    if isinstance(value, dict):
        return {key: without_notes(item) for key, item in value.items() if key != '_说明'}
    if isinstance(value, list):
        return [without_notes(item) for item in value]
    return value


def read_config(path):
    return without_notes(json.loads(Path(path).read_text(encoding='utf-8')))
