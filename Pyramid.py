"""
Builds an MP4 population-pyramid video from a diadporaPercent*.py CSV output.
Native and Immigrant stacked per single year of age (0-95), animated across
every year column in the file. Not a male/female pyramid (no sex split in
this model) -- age-structure bars, Native/Immigrant stacked, left-right
mirrored for the classic pyramid look.

Source data: whatever CSV path is given -- your own model output, not an
external data source. No citation applicable; this is a visualisation of
your own run.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

IN_PATH = "Population.csv"
OUT_PATH = "Pyramid.mp4"
FPS = 8
MAX_AGE = 95


def parse_num(s):
    s = s.strip().strip('"').replace(",", "")
    return float(s) if s not in ("", None) else 0.0


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    years = [int(y) for y in rows[1][2:]]
    native = np.zeros((MAX_AGE + 1, len(years)))
    immig = np.zeros((MAX_AGE + 1, len(years)))
    for r in rows[2:]:
        if len(r) < 2 or r[1] not in ("Native", "Immigrant"):
            continue
        age = int(r[0])
        vals = [parse_num(v) for v in r[2:2 + len(years)]]
        if r[1] == "Native":
            native[age] = vals
        else:
            immig[age] = vals
    return years, native, immig


def main():
    years, native, immig = load(IN_PATH)
    max_val = (native + immig).max()

    fig, ax = plt.subplots(figsize=(9, 7))
    ages = np.arange(MAX_AGE + 1)

    def draw(frame):
        ax.clear()
        n, im = native[:, frame], immig[:, frame]
        ax.barh(ages, n, color="#4C72B0", label="Native")
        ax.barh(ages, im, left=n, color="#DD8452", label="Immigrant")
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(-1, MAX_AGE + 1)
        ax.set_xlabel("Population (persons)")
        ax.set_ylabel("Age")
        ax.set_title(f"NZ Population Pyramid -- {years[frame]}")
        ax.legend(loc="upper right")
        ax.ticklabel_format(style="plain", axis="x")

    anim = FuncAnimation(fig, draw, frames=len(years), interval=1000 / FPS)
    writer = FFMpegWriter(fps=FPS, bitrate=2400)
    anim.save(OUT_PATH, writer=writer)
    print(f"Wrote {OUT_PATH}: {len(years)} frames at {FPS} fps")


if __name__ == "__main__":
    main()
