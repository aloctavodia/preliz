import ast
import inspect
import textwrap

import numpy as np

from preliz.distributions.distributions import Distribution


def is_pz_call(node, ns):
    """
    Detect if an AST node is a call to a preliz distribution.
    """
    if not isinstance(node, ast.Call):
        return False

    func = node.func

    # Case 1: Direct call (e.g., Normal(...))
    if isinstance(func, ast.Name):
        dist_cls = ns.get(func.id)
        if dist_cls is not None:
            try:
                return isinstance(dist_cls, type) and issubclass(dist_cls, Distribution)
            except TypeError:
                return False

    # Case 2: Attribute call (e.g., pz.Normal(...), preliz.Normal(...))
    elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        base = ns.get(func.value.id)
        if base is not None:
            dist_cls = getattr(base, func.attr, None)
            if dist_cls is not None:
                try:
                    return isinstance(dist_cls, type) and issubclass(dist_cls, Distribution)
                except TypeError:
                    return False

    return False


def add_rvs_to_pz_calls(node, ns, in_pz_call=False):
    """
    Recursively add .rvs() calls to outermost preliz distributions.
    """
    if isinstance(node, ast.Call):
        # Check if this is a pz call BEFORE recursing
        is_current_pz = is_pz_call(node, ns)

        # Recursively process func, args, keywords
        node.func = add_rvs_to_pz_calls(node.func, ns, in_pz_call)
        node.args = [
            add_rvs_to_pz_calls(arg, ns, True if is_current_pz else in_pz_call)
            for arg in node.args
        ]
        node.keywords = [
            ast.keyword(
                arg=kw.arg,
                value=add_rvs_to_pz_calls(kw.value, ns, True if is_current_pz else in_pz_call)
            )
            for kw in node.keywords
        ]

        # Add .rvs() only to outermost pz calls
        if is_current_pz and not in_pz_call:
            return ast.Call(
                func=ast.Attribute(value=node, attr='rvs', ctx=ast.Load()),
                args=[],
                keywords=[]
            )
        return node

    elif isinstance(node, (ast.List, ast.Tuple)):
        node.elts = [add_rvs_to_pz_calls(elt, ns, in_pz_call) for elt in node.elts]
        return node

    # For all other nodes, process fields recursively
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            new_list = [
                add_rvs_to_pz_calls(item, ns, in_pz_call) if isinstance(item, ast.AST) else item
                for item in value
            ]
            setattr(node, field, new_list)
        elif isinstance(value, ast.AST):
            setattr(node, field, add_rvs_to_pz_calls(value, ns, in_pz_call))

    return node


def sample(model_func, size=1000):
    """
    Sample from a generative model by automatically adding .rvs() calls
    to preliz distribution objects.
    """
    src = inspect.getsource(model_func)
    src = textwrap.dedent(src)
    tree = ast.parse(src)
    ns = model_func.__globals__.copy()

    # Transform the entire tree
    new_tree = add_rvs_to_pz_calls(tree, ns)

    ast.fix_missing_locations(new_tree)
    code = compile(new_tree, filename='<ast>', mode='exec')
    exec(code, ns)
    new_func = ns[model_func.__name__]

    return np.array([new_func() for _ in range(size)])
