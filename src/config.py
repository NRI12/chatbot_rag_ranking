"""Shared config loader: reads config.yaml and substitutes ${ENV_VAR} placeholders."""

import os
import re

import yaml
from dotenv import load_dotenv


def load_config(path: str = "config.yaml") -> dict:
    load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _substitute(raw)


def _substitute(obj):
    if isinstance(obj, str):
        def replacer(m):
            val = os.environ.get(m.group(1))
            if val is None:
                raise EnvironmentError(f"Environment variable '{m.group(1)}' is not set.")
            return val
        return re.sub(r"\$\{([^}]+)\}", replacer, obj)
    if isinstance(obj, dict):
        return {k: _substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(i) for i in obj]
    return obj
