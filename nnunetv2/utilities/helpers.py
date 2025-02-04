import torch

from nnunetv2.training.nnUNetTrainer.state import ExperimentState

def softmax_helper_dim0(x: torch.Tensor) -> torch.Tensor:
    if ExperimentState.mem_optimized:
        # Reimplementation of softmax function to use in place operation and save memory
        # previous_softmax = torch.softmax(x, 0)
        # print(torch.any(torch.isnan(previous_softmax)))
        # print(torch.any(torch.isinf(previous_softmax)))
        # print(torch.allclose(previous_softmax.float(), torch.softmax(x.float(), 0)))
        # print(torch.max(torch.abs(previous_softmax- torch.softmax(x-x.mean(), 0))))
        # # assert torch.allclose(previous_softmax, torch.softmax(x-x.mean(), 0), atol=1e-5)
        # # softmax(x) is equal to (softmax(x - max(x))) and avoids inf values
        # torch.exp(x - x.mean(), out=x)
        # # x is of shape [num_class, z, y, x]
        # step = 10
        # for i in torch.arange(0, x.shape[-1], step):
        #     index  = int(i)
        #     upto = min(index + step, x.shape[-1])
        #     x[:, :, :, index : upto] = torch.clamp(x[:, :, :, index : upto], min=torch.finfo(torch.float32).eps) / torch.clamp(torch.sum(
        #         x[:, :, :, index : upto], dim=0, keepdim=True
        #     ), min=torch.finfo(torch.float32).eps)

        # print(previous_softmax[2:4, 2:4, 2:4, 2:4])
        # print(x[2:4, 2:4, 2:4, 2:4])
        # print(torch.max(previous_softmax - x))
        # assert torch.allclose(x.half(),previous_softmax.half(), atol=torch.finfo(torch.float16).eps)

        # return x
        return torch.softmax(x, 0)
    else:
        return torch.softmax(x, 0)


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        from torch import mps
        mps.empty_cache()
    else:
        pass


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
