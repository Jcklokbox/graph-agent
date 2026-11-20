"""Agents under study and a loader.

An agent module exposes a class with the interface

    agent = Agent(n_inputs, n_actions, value_index, seed=..., **kw)
    action = agent.step(obs)          # obs: tuple of floats in [0, 1]
    agent.learn(next_obs)
    agent.predict_next(obs, action)   # optional: per-channel P(channel > 1/2)

`load("v11")` imports `agents.v11`; `load("path/to/file.py")` imports a file
(e.g. an archived version). The class is `Agent` if defined, else
`PosteriorGraph`.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys


def load(name: str):
    if name.endswith(".py"):
        mod_name = "agent_" + os.path.splitext(os.path.basename(name))[0]
        if mod_name in sys.modules:
            return sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(mod_name, name)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(f"agents.{name}")


def agent_class(mod):
    return getattr(mod, "Agent", None) or getattr(mod, "PosteriorGraph")
