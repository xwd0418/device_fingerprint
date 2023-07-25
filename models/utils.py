def convert_layers(model, layer_type_old, layer_type_new, convert_weights=False, num_groups=None):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_layers(module, layer_type_old, layer_type_new, convert_weights, num_groups)

        if type(module) == layer_type_old:
            layer_old = module
            layer_new = layer_type_new(module.num_features if num_groups is None else num_groups, module.num_features, module.eps, module.affine) 

            if convert_weights:
                layer_new.weight = layer_old.weight
                layer_new.bias = layer_old.bias

            model._modules[name] = layer_new
    return model