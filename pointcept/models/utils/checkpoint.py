"""
Checkpoint Utils for Models

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
from pointcept.utils.logger import get_root_logger

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


def load_checkpoint(model, filename, map_location="cpu", strict=False, logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Checkpoint file path.
        map_location (str): Same as in :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and checkpoint.
        logger (:obj:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    if logger is None:
        logger = get_root_logger()
    
    # torch.load를 사용하여 파일에서 체크포인트 로드
    checkpoint = torch.load(filename, map_location=map_location)
    
    # 체크포인트가 딕셔너리 형태인지 확인
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
        
    # state_dict 키가 있는지 확인
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # DDP(DistributedDataParallel)로 학습된 모델의 'module.' 접두사 제거
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # 모델에 state_dict 로드
    if hasattr(model, "load_state_dict"):
        msg = model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Loaded successfully from {filename}")
        logger.info(f"Missing keys: {msg.missing_keys}")
    else:
        logger.error(f"Target model has no 'load_state_dict' method. ")
    
    return checkpoint