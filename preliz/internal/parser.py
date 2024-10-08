import importlib
import inspect
import re
from sys import modules

import numpy as np

from preliz import distributions
from .distribution_helper import init_vals


def inspect_source(fmodel):
    source = inspect.getsource(fmodel)
    signature = inspect.signature(fmodel)
    source = re.sub(r"#.*$|^#.*$", "", source, flags=re.MULTILINE)
    default_params = {
        name: (param.default if param.default is not inspect.Parameter.empty else np.nan)
        for name, param in signature.parameters.items()
    }
    model = fmodel(**default_params)
    return source, signature, get_engine(model)


def get_engine(model):
    if getattr(model, "basic_RVs", False):
        return "pymc"
    elif getattr(model, "formula", False):
        return "bambi"
    return "preliz"


def parse_function_for_pred_textboxes(source, signature, engine="preliz"):
    model = {}

    slidify = list(signature.parameters.keys())
    regex = r"\b" + r"\b|\b".join(slidify) + r"\b"

    all_dist_str = dist_as_str()
    matches = match_preliz_dist(all_dist_str, source, engine)

    for match in matches:
        if engine == "bambi":
            dist_name_str = match.group(1)
        else:
            dist_name_str = match.group(2)
        if engine == "bambi":
            arguments = [s.strip() for s in match.group(2).split(",")]
        else:
            arguments = [s.strip() for s in match.group(3).split(",")]
        if engine == "pymc":
            args = pymc_parse_arguments(arguments, regex)
        else:
            args = parse_arguments(arguments, regex)
        for arg in args:
            if arg:
                func, var, idx = arg
                dist = getattr(distributions, dist_name_str)
                model[var] = (dist(**init_vals[dist_name_str]), idx, func)

    return model


def parse_arguments(lst, regex):
    result = []
    for idx, item in enumerate(lst):
        match = re.search(regex, item)
        if match:
            if item.isidentifier():
                result.append((None, match.group(0), idx))
            else:
                if "**" in item:
                    power = item.split("**")[1].strip()
                    result.append((power, match.group(0), idx))
                else:
                    func = item.split("(")[0].split(".")[-1]
                    result.append((func, match.group(0), idx))
    return result


def pymc_parse_arguments(lst, regex):
    result = []
    for idx, item in enumerate(lst):
        match = re.search(regex, item)
        if match:
            if item.isidentifier():
                result.append((None, match.group(0), idx - 1))
            else:
                if "**" in item:
                    power = item.split("**")[1].strip()
                    result.append((power, match.group(0), idx - 1))
                else:
                    func = item.split("(")[0].split(".")[-1]
                    result.append((func, match.group(0), idx - 1))
    return result


def get_prior_pp_samples(fmodel, variables, draws, engine=None, values=None):
    if values is None:
        values = []

    if engine == "preliz":
        obs_rv = variables[-1]  # only one observed for the moment
        pp_samples_ = []
        prior_samples_ = {name: [] for name in variables[:-1]}
        for _ in range(draws):
            for name, value in zip(variables, fmodel(*values)):
                if name == obs_rv:
                    pp_samples_.append(value)
                else:
                    prior_samples_[name].append(value)

        pp_samples = np.stack(pp_samples_)
        prior_samples = {key: np.array(val) for key, val in prior_samples_.items()}
    elif engine == "bambi":
        *prior_samples_, pp_samples = fmodel(*values)
        prior_samples = {name: np.array(val) for name, val in zip(variables[:-1], prior_samples_)}

    return pp_samples, prior_samples


def from_preliz(fmodel):
    source = inspect.getsource(fmodel)
    variables = match_return_variables(source)
    # Find the priors we want to change
    all_dist_str = dist_as_str()
    matches = match_preliz_dist(all_dist_str, source, "preliz")
    # Create a dictionary with the priors
    model = dict_model(matches, variables)

    return variables, model


def from_bambi(fmodel, draws):
    module_name = fmodel.__module__
    module = importlib.import_module(module_name)

    # Get the source code of the original function
    original_source = inspect.getsource(fmodel)

    # Define a pattern to find the line where the model is built
    pattern = re.compile(r"(\s+)([a-zA-Z_]\w*)\s*=\s*.*?Model(.*)")

    # Find the match in the source code
    match = pattern.search(original_source)

    # Extract the indentation and variable name
    indentation = match.group(1)
    variable_name = match.group(2)

    # Find the variables after the return statement
    return_variables = match_return_variables(original_source)

    if return_variables:
        # Build the new source code
        new_source = original_source.replace(
            match.group(0),
            f"{match.group(0)}"
            f"{indentation}{variable_name}.build()\n"
            f"{indentation}variables = [{variable_name}.backend.model.named_vars[v] "
            f"for v in {return_variables}]\n"
            f'{indentation}{", ".join(return_variables)} = pm.draw(variables, draws={draws})',
        )

        # Find the priors we want to change
        all_dist_str = dist_as_str()
        matches = match_preliz_dist(all_dist_str, new_source, "bambi")
        # Create a dictionary with the priors
        model = dict_model(matches, return_variables)

        # Execute the new source code to redefine the function
        exec(new_source, module.__dict__)  # pylint: disable=exec-used
        modified_fmodel = getattr(module, fmodel.__name__)

    return modified_fmodel, return_variables, model


def match_preliz_dist(all_dist_str, source, engine):
    # remove comments
    source = re.sub(r"#.*$|^#.*$", "", source, flags=re.MULTILINE)

    if engine in ["preliz", "pymc"]:
        regex = rf"(.*?({all_dist_str}).*?)\(([^()]*(?:\([^()]*\)[^()]*)*)\)"
    if engine == "bambi":
        regex = rf'\s*(?:\w+\.)?Prior\("({all_dist_str})",\s*((?:\w+=\w+(?:,?\s*)?)*)\s*\)'
    matches = re.finditer(regex, source)
    return matches


def match_return_variables(source):
    match = re.search(r"return (\w+(\s*,\s*\w+)*)", source)
    return [var.strip() for var in match.group(1).split(",")]


def dist_as_str():
    all_distributions = modules["preliz.distributions"].__all__
    return "|".join(all_distributions)


def dict_model(matches, return_variables):
    model = {}
    obs_rv = return_variables[-1]
    for match in matches:
        var_name = match.group(0).split("=")[0].strip()
        if var_name != obs_rv:
            dist = getattr(modules["preliz.distributions"], match.group(2))
            model[var_name] = dist()

    return model
