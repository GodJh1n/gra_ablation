#!/usr/bin/env python3
# plot_Tsub_pub.py — Publication-quality plot for c_Tsub vs time
# Requirements: numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_span(span_str, unit="ps"):
    """Parse 'a:b' into (a, b) floats; allow None if omitted."""
    if span_str is None:
        return None
    if ":" not in span_str:
        raise ValueError(f"Expected 'min:max' for {unit} span, got: {span_str}")
    a, b = span_str.split(":")
    a = float(a) if a.strip() else None
    b = float(b) if b.strip() else None
    return (a, b)

def parse_excludes(ex_str):
    """Parse 'a:b,c:d,...' into list of (a,b) floats for exclusion (in ps)."""
    if not ex_str:
        return []
    spans = []
    for chunk in ex_str.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        spans.append(parse_span(chunk))
    return spans

def moving_average(y, win):
    if win is None or win < 2:
        return y
    win = int(win)
    if win > len(y):
        win = len(y)
    kernel = np.ones(win) / win
    # 'same' keeps length; edges will be smoothed too
    return np.convolve(y, kernel, mode="same")

def load_temps(path):
    # Format per line: TimeStep c_Tmobile c_Tsub c_Tbeam
    data = np.loadtxt(path, comments="#")
    steps = data[:, 0]
    Tsub  = data[:, 2]
    return steps, Tsub

def main():
    ap = argparse.ArgumentParser(description="Plot graphene substrate temperature (c_Tsub) vs time (publication quality).")
    ap.add_argument("--file", default="./1000k_out/1000_temp/temps.out", help="path to temps.out")
    ap.add_argument("--dt-fs", type=float, default=0.1, help="LAMMPS timestep in femtoseconds (fs), default=0.1")
    ap.add_argument("--xlim", default='0:35', help="time range in ps, format 'min:max' (e.g., '5:40'). Leave blank for auto")
    ap.add_argument("--ylim", default='500:3000', help="temperature range in K, format 'min:max' (e.g., '1200:1600'). Leave blank for auto")
    ap.add_argument("--exclude", default=None, help="exclude time spans in ps, comma-separated 'a:b,c:d' (e.g., '0:5,42:50')")
    ap.add_argument("--smooth", type=int, default=0, help="moving-average window size (points). 0/1 = no smoothing")
    ap.add_argument("--marker-every", type=int, default=0, help="plot markers every N points (0=off)")
    ap.add_argument("--figsize", default="6,3.2", help="figure size in inches 'W,H' (e.g., '6,3.2')")
    ap.add_argument("--font", type=int, default=11, help="base font size")
    ap.add_argument("--dpi", type=int, default=300, help="save DPI")
    ap.add_argument("--out", default="Tsub_vs_time.png", help="output image filename")
    args = ap.parse_args()
    plt.show()

    # Load
    steps, Tsub = load_temps(args.file)
    time_ps = steps * args.dt_fs / 1000.0

    # Build inclusion mask (start from all True)
    mask = np.ones_like(time_ps, dtype=bool)

    # Apply x-range (view window)
    xspan = parse_span(args.xlim) if args.xlim else (None, None)
    if xspan is not None:
        xmin, xmax = xspan
        if xmin is not None:
            mask &= (time_ps >= xmin)
        if xmax is not None:
            mask &= (time_ps <= xmax)

    # Exclude spans
    for span in parse_excludes(args.exclude):
        a, b = span
        if a is None or b is None:
            continue
        # remove a≤t≤b
        mask &= ~((time_ps >= a) & (time_ps <= b))

    t = time_ps[mask]
    y = Tsub[mask]

    # Optional smoothing (moving average)
    y_smooth = moving_average(y, args.smooth)

    # Figure / fonts
    W, H = (float(x) for x in args.figsize.split(","))
    plt.rcParams.update({
        "font.size": args.font,
        "axes.labelsize": args.font,
        "axes.titlesize": args.font + 1,
        "xtick.labelsize": args.font,
        "ytick.labelsize": args.font,
        "legend.fontsize": args.font,
        "figure.dpi": args.dpi,
    })

    fig, ax = plt.subplots(figsize=(W, H), constrained_layout=True)

    # Plot raw
    kw = {"linewidth": 1.7, "alpha": 0.9}
    if args.marker_every and args.marker_every > 0:
        kw.update({"marker": "o", "markersize": 2.8, "markevery": args.marker_every})
    raw_line = ax.plot(t, y, label="c_Tsub (raw)", **kw)

    # Plot smoothed (different linestyle; color will auto-cycle to distinguish)
    if args.smooth and args.smooth > 1:
        smooth_line = ax.plot(t, y_smooth, linestyle="--", linewidth=1.8, label=f"MA (win={int(args.smooth)})")

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Substrate temperature, c_Tsub (K)")
    ax.set_title("Graphene substrate temperature vs time")

    # Y limits
    if args.ylim:
        ymin, ymax = parse_span(args.ylim, unit="K")
        if ymin is not None or ymax is not None:
            ax.set_ylim((ymin if ymin is not None else ax.get_ylim()[0],
                         ymax if ymax is not None else ax.get_ylim()[1]))

    # Nice grid and spine cosmetics
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(frameon=False)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
