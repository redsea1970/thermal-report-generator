import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    with open("report.json", "r", encoding="utf-8") as f:
        rep = json.load(f)
    thermal = np.load("sample_thermal.npy")
    annotated_bgr = cv2.imread("thermal_annotated.png")
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    zones = rep["zones"]
    ids = [f"#{z['id']} {z['severity']}" for z in zones]
    tmax = [z["t_max"] for z in zones]
    sev_color = {"CRITICAL": "#e53935", "HIGH": "#fb8c00",
                 "MEDIUM": "#fdd835", "LOW": "#43a047"}
    colors = [sev_color[z["severity"]] for z in zones]

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.imshow(annotated_rgb)
    ax1.set_title("Thermal inspection - annotated")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(ids, tmax, color=colors)
    ax2.axhline(60, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("t_max, C")
    ax2.set_title("Hot zones by peak temperature")
    ax2.tick_params(axis="x", rotation=15)

    ax3 = fig.add_subplot(gs[1, 1])
    im = ax3.imshow(thermal, cmap="inferno")
    ax3.set_title("Thermal field (raw)")
    ax3.axis("off")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="C")

    fig.tight_layout()
    fig.savefig("thermal_report.png", dpi=130)
    print("Saved thermal_report.png")

if __name__ == "__main__":
    main()
