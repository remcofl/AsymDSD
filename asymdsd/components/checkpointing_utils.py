from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_module_from_checkpoint(
    ckpt_path: str | Path,
    module: nn.Module,
    device: torch.device | None = None,
    key_prefix: str | list[str] | None = None,
    replace_key_part: dict[str, str] | None = None,
    strict: bool = True,
) -> None:
    """
    Load a module from a checkpoint.

    Args:
        ckpt_path: Path to the checkpoint. Loaded using torch.load.
        module: Module to load the checkpoint into.
        device: Device to load the checkpoint to.
        key_prefix: Prefix of state_dict keys to select subset from the checkpoint.
            If None, the entire checkpoint will be loaded.
            E.g., if the checkpoint has keys ['a.b', 'a.c', 'd'], and key_prefix=['a'],
            then the loaded checkpoint will have keys ['b', 'c'].
    """

    state_dict: dict[str, Any] = torch.load(ckpt_path, map_location=device)

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if key_prefix is None:
        # Simply load based on the entire state_dict
        module.load_state_dict(state_dict, strict=strict)
        return

    if isinstance(key_prefix, str):
        key_prefix = [key_prefix]

    module_state_dict = {}

    for state_key, value in state_dict.items():
        for key in key_prefix:
            key_dot = f"{key}."
            if state_key.startswith(key_dot):
                updated_state_key = state_key.replace(key_dot, "")
                if replace_key_part is not None:
                    for old_key, new_key in replace_key_part.items():
                        updated_state_key = updated_state_key.replace(old_key, new_key)
                module_state_dict[updated_state_key] = value
                break

    module.load_state_dict(module_state_dict, strict=strict)
