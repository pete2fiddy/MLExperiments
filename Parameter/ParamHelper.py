

def filter_non_default_params(params, def_params):
    out_params = def_params.copy()
    for param in params.keys():
        out_params[param] = params[param]
    print("out params; ", out_params)
    return out_params
