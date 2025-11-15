import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kernels import matmul_int4_fused, quantize_rowwise


def load_llama_model(device: str = "cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
    model = model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return tokenizer, model


def get_model_size_mb(model: torch.nn.Module) -> float:
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


class QuantLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_quant: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_q = weight_quant
        self.weight_scale = weight_scale
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        y_2d = matmul_int4_fused(
            x=x_2d,
            w_q=self.weight_q,
            w_scale=self.weight_scale,
            bias=self.bias,
        )
        y = y_2d.reshape(*orig_shape[:-1], self.out_features)
        return y


def quantize_linear_module(linear: torch.nn.Linear, device: torch.device | str | None = None) -> QuantLinear:
    if device is None:
        device = next(linear.parameters()).device
    in_features = linear.in_features
    out_features = linear.out_features
    weight = linear.weight.detach().to(device=device, dtype=torch.float16)
    weight_q, max_values = quantize_rowwise(weight)
    bias = None
    if linear.bias is not None:
        bias = linear.bias.detach().to(device=device, dtype=torch.float16)
    return QuantLinear(in_features, out_features, weight_q, max_values, bias)


def replace_linear_with_quantlinear(module: torch.nn.Module, device: torch.device | str | None = None):
    for child_name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            quant_linear = quantize_linear_module(child, device=device)
            setattr(module, child_name, quant_linear)
        else:
            replace_linear_with_quantlinear(child, device=device)


