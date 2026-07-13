"""Shared plotting utilities."""

from matplotlib.ticker import FuncFormatter, MaxNLocator


def format_tick(value, _pos):
    """Drop trailing '.0' from integer values, keep minimal decimals otherwise."""
    return f"{int(value)}" if float(value).is_integer() else f"{value:g}"


def apply_square_ticks(ax, size_x, size_y):
    """Set identical tick positions on x and y axes, with clean integer labels."""
    size = max(size_x, size_y)
    ticks = MaxNLocator(nbins = "auto", steps = [1, 2, 2.5, 5, 10]).tick_values(0, size)
    ticks = [t for t in ticks if 0 <= t <= size]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(format_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(format_tick))
