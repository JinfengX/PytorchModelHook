from typing import Union, List

from torch import nn
import torch
from torch.nn import Conv2d, Linear, AdaptiveAvgPool2d


class ModelHook:
    def __init__(
        self,
        register_module_name: Union[List[str], str],
        hook_and_action: Union[List[str], str],
    ):
        """Register hooks for model

        Args:
            register_module_name: the name of the module to be registered
            hook_and_action: the type of hook to be registered and corresponding action,
                the type of hook including 'forward', 'backward', the actions including 'getInput', 'getOutput',
                use '_' to connect the type of hook and action, e.g. 'forward_getInput', 'backward_getOutput'
        """
        if isinstance(register_module_name, list) and isinstance(hook_and_action, list):
            assert len(register_module_name) == len(hook_and_action)
        else:
            assert isinstance(register_module_name, str) and isinstance(
                hook_and_action, str
            )
            register_module_name, hook_and_action = [register_module_name], [
                hook_and_action
            ]
        self.model_name = "Model"
        self.register_module_name = register_module_name
        self.hook, self.action = [], []
        for h_a in hook_and_action:
            h, a = h_a.split("_")
            self.hook.append(self.hook_types(h))
            self.action.append(self.action_types(a))
        self.module_hooks = {
            name: (hook, action)
            for name, hook, action in zip(
                self.register_module_name, self.hook, self.action
            )
        }
        self.output = dict.fromkeys(self.register_module_name)

    def hook_types(self, name):
        if name == "forward":
            return "register_forward_hook"
        elif name == "backward":
            return "register_backward_hook"

    def action_types(self, name):

        def get_input(module_name):
            # Store input with module name
            def _get_input(module, input, output):
                if isinstance(input, tuple) and len(input) == 1:
                    self.output[module_name] = input[0]
                else:
                    # If there are multiple input tensors, raise an error
                    raise NotImplementedError(
                        "Multiple input tensors are not supported yet"
                    )

            return _get_input

        def get_output(module_name):
            # Store output with module name
            def _get_output(module, input, output):
                self.output[module_name] = output

            return _get_output

        if name == "getInput":
            return get_input
        elif name == "getOutput":
            return get_output
        else:
            raise ValueError("Invalid hook action")

    def register_hooks(self, model):
        for name, module in model.named_modules():
            if name in self.register_module_name:
                hook, action = self.module_hooks[name]
                getattr(module, hook)(action(name))
        self.model_name = model.__class__.__name__

    def __str__(self):
        str = f"Hooks of {self.model_name}: \n"
        for name, output in self.output.items():
            str += f"{name}: {output.shape}\n"
        return str


# test Model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.layer1 = Linear(in_features=32, out_features=64)
        self.avgpool = AdaptiveAvgPool2d(1)

    def forward(self, x):
        conv1_out = self.conv1(x)
        avgpool_out = self.avgpool(conv1_out)
        flat = torch.flatten(avgpool_out, 1)
        layer1_out = self.layer1(flat)
        return {
            "conv1_out": conv1_out,
            "avgpool_out": avgpool_out,
            "flat": flat,
            "layer1_out": layer1_out,
        }


if __name__ == "__main__":
    print("start test")

    model = TestModel()
    print(model)

    model_hook = ModelHook("layer1", "forward_getOutput")
    model_hook.register_hooks(model)
    model_hook_multi = ModelHook(
        ["conv1", "layer1"], ["forward_getOutput", "forward_getInput"]
    )
    model_hook_multi.register_hooks(model)

    x = torch.randn([32, 3, 224, 224])
    model_out = model(x)

    assert torch.equal(model_hook.output["layer1"], model_out["layer1_out"])
    assert torch.equal(model_hook_multi.output["conv1"], model_out["conv1_out"])
    assert torch.equal(model_hook_multi.output["layer1"], model_out["flat"])

    print(f"capture intermediate layer:\n{model_hook}")
    print(f"capture multi intermediate layer:\n{model_hook_multi}")

    print("end of test")
