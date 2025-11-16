# python import
from typing import Union
# package import
# local import

__all__ = ["get_unique_attr_across"]


def get_unique_attr_across(modules: list, attr: Union[str, list, dict]):
    results = []
    if isinstance(attr, dict):
        for func_name, params in attr.items():
            funcs = [getattr(module, func_name, None) for module in modules]
            valid_funcs = [func for func in funcs if func is not None]
            if not valid_funcs:
                raise AttributeError(f"Attribute '{func_name}' not found in any of the modules: {str(modules)}")
            elif len(valid_funcs) > 1:
                raise AttributeError(f"Attribute '{func_name}' found in multiple modules: {str(modules)}")
            else:
                func = valid_funcs[0]
                # if failed, check the value of params
                if params is None:
                    result = func()
                elif isinstance(params, dict):
                    result = func(**params)
                else:
                    result = func(params)
                results.append(result)
    elif isinstance(attr, list):
        for func_name in attr:
            funcs = [getattr(module, func_name, None) for module in modules]
            valid_funcs = [func for func in funcs if func is not None]
            if not valid_funcs:
                raise AttributeError(f"Attribute '{func_name}' not found in any of the modules: {str(modules)}")
            elif len(valid_funcs) > 1:
                raise AttributeError(f"Attribute '{func_name}' found in multiple modules: {str(modules)}")
            else:
                results += valid_funcs
    else:
        assert isinstance(attr, str), "attr must be type of dict, list or str."
        funcs = [getattr(module, attr, None) for module in modules]
        valid_funcs = [func for func in funcs if func is not None]
        if not valid_funcs:
            raise AttributeError(f"Attribute '{attr}' not found in any of the modules: {str(modules)}")
        elif len(valid_funcs) > 1:
            raise AttributeError(f"Attribute '{attr}' found in multiple modules: {str(modules)}")
        else:
            results = valid_funcs[0]
    return results
