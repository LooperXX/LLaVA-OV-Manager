import os
from .manager import StaticManager, AdaptiveManager, StaticManagerZeroInit, AdaptiveManagerZeroInit
from llava.utils import rank0_print
from torch import nn

def build_manager(config):
    num_hidden_layers_llm = config.num_hidden_layers
    hidden_size = config.hidden_size

    assert num_hidden_layers_llm > 0, "num_hidden_layers_llm is invalid"
    assert hidden_size > 0, "hidden_size is invalid"

    mm_manager_type = getattr(config, "mm_manager_type", None)
    mm_manager_vision_select_layers_start = getattr(config, "mm_manager_vision_select_layers_start", -1)
    mm_manager_vision_select_layers_interval = getattr(config, "mm_manager_vision_select_layers_interval", -1)
    mm_manager_vision_select_layers_end= getattr(config, "mm_manager_vision_select_layers_end", -1)
    mm_manager_injection_start = getattr(config, "mm_manager_injection_start", -1)
    mm_manager_injection_interval = getattr(config, "mm_manager_injection_interval", -1)
    mm_manager_injection_end = getattr(config, "mm_manager_injection_end", -1)
    mm_manager_residual = getattr(config, "mm_manager_residual", False)

    assert mm_manager_type is not None, "mm_manager_type is required"
    assert mm_manager_vision_select_layers_start >= 0, "mm_manager_vision_select_layers_start is required"
    assert mm_manager_vision_select_layers_interval >= 1, "mm_manager_vision_select_layers_interval is required"
    assert mm_manager_vision_select_layers_end >= 1, "mm_manager_vision_select_layers_end is required"
    assert mm_manager_injection_start >= 0, "mm_manager_injection_start is required"
    assert mm_manager_injection_interval >= 1, "mm_manager_injection_interval is required"
    assert mm_manager_injection_end >= 1, "mm_manager_injection_end is required"

    mm_manager_select_layer_index_list = list(range(mm_manager_vision_select_layers_start, mm_manager_vision_select_layers_end, mm_manager_vision_select_layers_interval))
    num_manage_select_layer = len(mm_manager_select_layer_index_list)
    mm_manager_index_list = list(range(mm_manager_injection_start, min(num_hidden_layers_llm, mm_manager_injection_end), mm_manager_injection_interval))

    rank0_print(f"Building Manager: {mm_manager_type}")
    rank0_print(f"Vision select layer index: {mm_manager_select_layer_index_list}")
    rank0_print(f"# of manage select layers: {num_manage_select_layer}")
    rank0_print(f"Manager injection index: {mm_manager_index_list}")
    rank0_print(f"# of manage layers: {len(mm_manager_index_list)}")
    rank0_print(f"Manager residual: {mm_manager_residual}")

    assert mm_manager_residual is True, "Zero init manager only supports residual type manager"
    if "static" in mm_manager_type:
        return nn.ModuleList([StaticManagerZeroInit(config, num_manage_select_layer=num_manage_select_layer, layer_index=i, hidden_size=hidden_size) for i in mm_manager_index_list]), mm_manager_select_layer_index_list, mm_manager_index_list
    elif "adaptive" in mm_manager_type:
        if 0 in mm_manager_index_list:
            return nn.ModuleList([StaticManagerZeroInit(config, num_manage_select_layer=num_manage_select_layer, layer_index=0, hidden_size=hidden_size)] + [AdaptiveManagerZeroInit(config, num_manage_select_layer=num_manage_select_layer, layer_index=i, hidden_size=hidden_size) for i in mm_manager_index_list[1:]]), mm_manager_select_layer_index_list, mm_manager_index_list
        else:
            return nn.ModuleList([AdaptiveManagerZeroInit(config, num_manage_select_layer=num_manage_select_layer, layer_index=i, hidden_size=hidden_size) for i in mm_manager_index_list]), mm_manager_select_layer_index_list, mm_manager_index_list

    raise ValueError(f"Unknown manager type: {mm_manager_type}")
