# PytorchModelHook
A hook tool for capturing the Intermediate layer features without changing the code of original model.

```
model = Model() # define your model

"""
The ModelHook class takes two parameters:(register_module_name, hook_and_action)
register_module_name is the name of the module(s) to be registered. It can be a single module name or a list of module names.
hook_and_action is the type of hook to be registered and the corresponding action(s).
If a single module is registered, hook_and_action should be a string indicating the type of hook and action to be registered.
If multiple modules are being registered, hook_and_action should be a list of strings or sublists, where each sublist contains the type of hook and action(s) for each module.
The hook types include 'forward' and 'backward', and the actions include 'getInput', 'getOutput', 'getInputGrad' and 'getOutputGrad'.
Use '_' to connect the hook types and actions, e.g. 'forward_getInput', 'backward_getOutput'.
"""
model_hook = ModelHook("layer1", "forward_getOutput") # capture single layer features
model_hook.register_hooks(model) 
model_hook_multi = ModelHook(
    ["conv1", "layer1"], ["forward_getOutput", "forward_getInput"]
) # capture multi-layer features
model_hook_multi.register_hooks(model)
model_hook_multiAction = ModelHook(
    ["conv1", "layer1"],
    [
        [
            "forward_getInput",
            "forward_getOutput",
            "backward_getInputGrad",
            "backward_getOutputGrad",
        ],
        [
            "forward_getInput",
            "forward_getOutput",
            "backward_getInputGrad",
            "backward_getOutputGrad",
        ],
    ],
) # capture multi-features corresponding to multi-layer 
model_hook_multiAction.register_hooks(model)

x = torch.randn([32, 3, 224, 224]) # model input
model_out = model(x) # model forward
loss = model_out["layer1_out"].sum() # loss
loss.backward() # model backward

# validate the captured features
assert torch.equal(
    model_hook["layer1"]["forward_getOutput"], model_out["layer1_out"]
)
print(f"capture intermediate layer:\n{model_hook}")
    
assert torch.equal(
    model_hook_multi["conv1"]["forward_getOutput"], model_out["conv1_out"]
)
assert torch.equal(
    model_hook_multi["layer1"]["forward_getInput"], model_out["flat"]
)
print(f"capture multi intermediate layer:\n{model_hook_multi}")

# remove the hook functions after usage
model_hook.remove_hooks()
model_hook_multi.remove_hooks()
model_hook_multiAction.remove_hooks()

print("end of test")
```
