import numpy as np
import cv2

def generate_thermal(width=640, height=480, seed=42):
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=20.0, scale=2.0, size=(height, width)).astype(np.float32)

    yy, xx = np.mgrid[0:height, 0:width]

    def hot_spot(cx, cy, r, peak):
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        return peak * np.exp(-d2 / (2.0 * r * r))

    spots = [
        (160, 140, 40, 65.0),
        (420, 220, 55, 80.0),
        (520, 380, 30, 55.0),
        (260, 360, 35, 45.0),
    ]
    thermal = base.copy()
    for cx, cy, r, peak in spots:
        thermal += hot_spot(cx, cy, r, peak)

    noise = rng.normal(loc=0.0, scale=0.5, size=(height, width)).astype(np.float32)
    thermal += noise
    return np.clip(thermal, -10.0, 120.0)

def to_colormap(thermal, tmin=15.0, tmax=90.0):
    norm = np.clip((thermal - tmin) / (tmax - tmin), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

if __name__ == "__main__":
    t = generate_thermal()
    img = to_colormap(t)
    cv2.imwrite("sample_thermal.png", img)
    np.save("sample_thermal.npy", t)
    print(f"Saved sample_thermal.png and .npy | min={t.min():.1f}C max={t.max():.1f}C mean={t.mean():.1f}C")
