import inspect
import logging
import traceback
import sys

from IPython import get_ipython
from ipywidgets import FloatSlider, IntSlider
from arviz import plot_kde, plot_ecdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers, get_backend
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.interpolate import interp1d, PchipInterpolator

_log = logging.getLogger("preliz")


def plot_pointinterval(distribution, quantiles=None, rotated=False, ax=None):
    """
    By default plot the quantiles [0.05, 0.25, 0.5, 0.75, 0.95]
    The median as dot and the two interquantiles ranges (0.05-0.95) and (0.25-0.75) as lines.

    Parameters
    ----------
    distribution : preliz distribution or array
    quantiles : list
        The number of elements should be 5, 3, 1 or 0 (in this last case nothing will be plotted).
        defaults to [0.05, 0.25, 0.5, 0.75, 0.95].
    rotated : bool
        Whether to do the plot along the x-axis (default) or on the y-axis
    ax : matplotlib axis
    """
    if quantiles is None:
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    if isinstance(distribution, rv_frozen):
        q_s = distribution.ppf(quantiles).tolist()
    else:
        q_s = np.quantile(distribution, quantiles).tolist()

    q_s_size = len(q_s)
    if not q_s_size in (5, 3, 1, 0):
        raise ValueError("quantiles should have 5, 3, 1 or 0 elements")

    if rotated:
        if q_s_size == 5:
            ax.plot([0, 0], (q_s.pop(0), q_s.pop(-1)), "k", solid_capstyle="butt", lw=1.5)
        if q_s_size > 2:
            ax.plot([0, 0], (q_s.pop(0), q_s.pop(-1)), "k", solid_capstyle="butt", lw=4)
        if q_s_size > 0:
            ax.plot(0, q_s[0], "wo", mec="k")
    else:
        if q_s_size == 5:
            ax.plot((q_s.pop(0), q_s.pop(-1)), [0, 0], "k", solid_capstyle="butt", lw=1.5)
        if q_s_size > 2:
            ax.plot((q_s.pop(0), q_s.pop(-1)), [0, 0], "k", solid_capstyle="butt", lw=4)

        if q_s_size > 0:
            ax.plot(q_s[0], 0, "wo", mec="k")


def plot_pdfpmf(dist, moments, pointinterval, quantiles, support, legend, figsize, ax):
    ax = get_ax(ax, figsize)
    color = next(ax._get_lines.prop_cycler)["color"]
    if legend:
        label = repr_to_matplotlib(dist)

        if moments is not None:
            label += get_moments(dist, moments)

        if legend == "title":
            ax.set_title(label)
            label = None
    else:
        label = None

    x = dist.xvals(support)
    if dist.kind == "continuous":
        density = dist.pdf(x)
        ax.axhline(0, color="0.8", ls="--", zorder=0)
        ax.plot(x, density, label=label, color=color)
        ax.set_yticks([])
    else:
        mass = dist.pdf(x)
        eps = np.clip(dist._finite_endpoints(support), *dist.support)
        x_c = np.linspace(*eps, 1000)

        if len(x) > 2:
            interp = PchipInterpolator(x, mass)
        else:
            interp = interp1d(x, mass)

        mass_c = np.clip(interp(x_c), np.min(mass), np.max(mass))

        ax.axhline(0, color="0.8", ls="--", zorder=0)
        ax.plot(x_c, mass_c, ls="dotted", color=color)
        ax.plot(x, mass, "o", label=label, color=color)

    if pointinterval:
        plot_pointinterval(dist.rv_frozen, quantiles=quantiles, ax=ax)

    if legend == "legend":
        side_legend(ax)

    return ax


def plot_cdf(dist, moments, pointinterval, quantiles, support, legend, figsize, ax):
    ax = get_ax(ax, figsize)
    color = next(ax._get_lines.prop_cycler)["color"]
    if legend:
        label = repr_to_matplotlib(dist)

        if moments is not None:
            label += get_moments(dist, moments)

        if legend == "title":
            ax.set_title(label)
            label = None
    else:
        label = None

    eps = dist._finite_endpoints(support)
    x = np.linspace(*eps, 1000)
    cdf = dist.cdf(x)
    ax.plot(x, cdf, label=label, color=color)

    if pointinterval:
        plot_pointinterval(dist.rv_frozen, quantiles=quantiles, ax=ax)

    if legend == "legend":
        side_legend(ax)
    return ax


def plot_ppf(dist, moments, pointinterval, quantiles, legend, figsize, ax):
    ax = get_ax(ax, figsize)
    color = next(ax._get_lines.prop_cycler)["color"]

    if legend:
        label = repr_to_matplotlib(dist)

        if moments is not None:
            label += get_moments(dist, moments)

        if legend == "title":
            ax.set_title(label)
            label = None
    else:
        label = None

    x = np.linspace(0, 1, 1000)
    ax.plot(x, dist.ppf(x), label=label, color=color)

    if pointinterval:
        plot_pointinterval(dist.rv_frozen, quantiles=quantiles, rotated=True, ax=ax)

    if legend == "legend":
        side_legend(ax)
    return ax


def get_ax(ax, figsize):
    if ax is None:
        fig_manager = _pylab_helpers.Gcf.get_active()
        if fig_manager is not None:
            ax = fig_manager.canvas.figure.gca()
        else:
            _, ax = plt.subplots(figsize=figsize)
    return ax


def side_legend(ax):
    bbox = ax.get_position()
    ax.set_position([bbox.x0, bbox.y0, bbox.width, bbox.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))


def repr_to_matplotlib(distribution):
    string = distribution.__repr__()
    string = string.replace("\x1b[1m", r"$\bf{")
    string = string.replace("\x1b[0m", "}$")
    return string


def get_moments(dist, moments):
    names = {
        "m": "μ",
        "d": "σ",
        "s": "γ",
        "v": "σ²",
        "k": "κ",
    }
    str_m = []
    seen = []
    for moment in moments:
        if moment not in seen:
            if moment == "d":
                value = dist.rv_frozen.stats("v") ** 0.5
            else:
                value = dist.rv_frozen.stats(moment)
            if isinstance(value, (np.ndarray, int, float)):
                str_m.append(f"{names[moment]}={value:.3g}")

        seen.append(moment)

    return "\n" + ", ".join(str_m)


def get_slider(name, value, lower, upper, continuous_update=True):
    if np.isfinite(lower):
        min_v = lower
    else:
        min_v = value - 10
    if np.isfinite(upper):
        max_v = upper
    else:
        max_v = value + 10

    if isinstance(value, float):
        slider_type = FloatSlider
        step = (max_v - min_v) / 100
    else:
        slider_type = IntSlider
        step = 1

    slider = slider_type(
        min=min_v,
        max=max_v,
        step=step,
        description=f"{name} ({lower:.0f}, {upper:.0f})",
        value=value,
        style={"description_width": "initial"},
        continuous_update=continuous_update,
    )

    return slider


def get_sliders(signature, model):
    sliders = {}
    for name, param in signature.parameters.items():
        if isinstance(param.default, (int, float)):
            value = float(param.default)
        else:
            value = None

        dist, idx, func = model[name]
        lower, upper = dist.params_support[idx]
        # ((eps, 1 - eps),
        # (-np.inf, np.inf),
        # (eps, np.inf))
        # ((-np.pi, np.pi)
        if func is not None and lower == np.finfo(float).eps:
            if func in ["exp", "abs", "expit", "logistic"]:
                lower = -np.inf
            elif func in ["log"]:
                lower = 1.0
            elif func.replace(".", "", 1).isdigit():
                if not float(func) % 2:
                    lower = -np.inf

        if func is not None and upper == 1 - np.finfo(float).eps:
            if func in ["expit", "logistic"]:
                lower = np.inf

        if value is None:
            value = getattr(dist, dist.param_names[idx])

        sliders[name] = get_slider(name, value, lower, upper, continuous_update=False)
    return sliders


def plot_decorator(func, iterations, kind_plot):
    def looper(*args, **kwargs):
        results = []
        alpha = max(0.01, 1 - iterations * 0.009)
        for _ in range(iterations):
            result = func(*args, **kwargs)
            results.append(result)
            if kind_plot == "hist":
                plt.hist(
                    result, alpha=alpha, density=True, bins="auto", color="C0", histtype="step"
                )
            elif kind_plot == "kde":
                plot_kde(result, plot_kwargs={"alpha": alpha})
            elif kind_plot == "ecdf":
                plt.plot(
                    np.sort(result), np.linspace(0, 1, len(result), endpoint=False), color="C0"
                )

        if kind_plot == "hist":
            plt.hist(
                np.concatenate(results),
                density=True,
                bins="auto",
                color="k",
                ls="--",
                histtype="step",
            )
        elif kind_plot == "kde":
            plot_kde(np.concatenate(results), plot_kwargs={"color": "k", "ls": "--"})
        elif kind_plot == "ecdf":
            a = np.concatenate(results)
            plt.plot(np.sort(a), np.linspace(0, 1, len(a), endpoint=False), "k--")

    return looper


def plot_pp_samples(pp_samples, pp_samples_idxs, references, kind="pdf", sharex=True, fig=None):
    row_colum = int(np.ceil(len(pp_samples_idxs) ** 0.5))

    if fig is None:
        fig, axes = plt.subplots(row_colum, row_colum, figsize=(8, 6), constrained_layout=True)
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
    else:
        axes = np.array(fig.axes)

    try:
        axes = axes.ravel()
    except AttributeError:
        axes = [axes]

    x_lims = [np.inf, -np.inf]

    for ax, idx in zip(axes, pp_samples_idxs):
        ax.clear()
        for ref in references:
            ax.axvline(ref, ls="--", color="0.5")
        ax.relim()

        sample = pp_samples[idx]

        if sharex:
            min_ = sample.min()
            max_ = sample.max()
            if min_ < x_lims[0]:
                x_lims[0] = min_
            if max_ > x_lims[1]:
                x_lims[1] = max_

        if kind == "pdf":
            plot_kde(sample, ax=ax, plot_kwargs={"color": "C0"})  # pylint:disable=no-member
        elif kind == "hist":
            bins, *_ = ax.hist(
                sample, color="C0", bins="auto", alpha=0.5, density=True
            )  # pylint:disable=no-member
            ax.set_ylim(-bins.max() * 0.05, None)

        elif kind == "ecdf":
            plot_ecdf(sample, ax=ax, plot_kwargs={"color": "C0"})  # pylint:disable=no-member

        plot_pointinterval(sample, ax=ax)
        ax.set_title(idx, alpha=0)
        ax.set_yticks([])

    if sharex:
        for ax in axes:
            ax.set_xlim(np.floor(x_lims[0]), np.ceil(x_lims[1]))

    fig.canvas.draw()
    return fig, axes


def plot_pp_mean(pp_samples, selected, references=None, kind="pdf", fig_pp_mean=None):
    if fig_pp_mean is None:
        fig_pp_mean, ax_pp_mean = plt.subplots(1, 1, figsize=(8, 2), constrained_layout=True)
        fig_pp_mean.canvas.header_visible = False
        fig_pp_mean.canvas.footer_visible = False
    else:
        ax_pp_mean = fig_pp_mean.axes[0]

    ax_pp_mean.clear()

    if np.any(selected):
        sample = pp_samples[selected].ravel()
    else:
        sample = pp_samples.ravel()

    for ref in references:
        ax_pp_mean.axvline(ref, ls="--", color="0.5")

    if kind == "pdf":
        plot_kde(
            sample, ax=ax_pp_mean, plot_kwargs={"color": "k", "linestyle": "--"}
        )  # pylint:disable=no-member
    elif kind == "hist":
        bins, *_ = ax_pp_mean.hist(
            sample, color="k", ls="--", bins="auto", alpha=0.5, density=True
        )  # pylint:disable=no-member
        ax_pp_mean.set_ylim(-bins.max() * 0.05, None)

    elif kind == "ecdf":
        plot_ecdf(
            sample, ax=ax_pp_mean, plot_kwargs={"color": "k", "linestyle": "--"}
        )  # pylint:disable=no-member

    plot_pointinterval(sample, ax=ax_pp_mean)
    ax_pp_mean.set_yticks([])
    fig_pp_mean.canvas.draw()

    return fig_pp_mean


def check_inside_notebook(need_widget=False):
    shell = get_ipython()
    name = inspect.currentframe().f_back.f_code.co_name
    try:
        if shell is None:
            raise RuntimeError(
                f"To run {name}, you need to call it from within a Jupyter notebook or Jupyter lab."
            )
        if need_widget:
            shell_name = shell.__class__.__name__
            if shell_name == "ZMQInteractiveShell" and "nbagg" not in get_backend():
                msg = f"To run {name}, you need use the magic `%matplotlib widget`"
                raise RuntimeError(msg)
    except Exception:  # pylint: disable=broad-except
        tb_as_str = traceback.format_exc()
        # Print only the last line of the traceback, which contains the error message
        print(tb_as_str.strip().rsplit("\n", maxsplit=1)[-1], file=sys.stdout)