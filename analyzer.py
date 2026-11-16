import json
import numpy as np
import cv2

def find_hot_zones(thermal, threshold_c=60.0, min_area=150):
    mask = (thermal >= threshold_c).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zones = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = thermal[y:y + h, x:x + w]
        zones.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "area_px": int(area),
            "t_max": round(float(roi.max()), 2),
            "t_mean": round(float(roi.mean()), 2),
            "t_min": round(float(roi.min()), 2),
        })

    zones.sort(key=lambda z: z["t_max"], reverse=True)
    for i, z in enumerate(zones, start=1):
        z["id"] = i
        z["severity"] = classify(z["t_max"])
    return zones, mask

def classify(t_max):
    if t_max >= 90.0:
        return "CRITICAL"
    if t_max >= 75.0:
        return "HIGH"
    if t_max >= 60.0:
        return "MEDIUM"
    return "LOW"

def annotate(thermal, zones, tmin=15.0, tmax=90.0):
    norm = np.clip((thermal - tmin) / (tmax - tmin), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    img = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

    sev_color = {
        "CRITICAL": (0, 0, 255),
        "HIGH": (0, 128, 255),
        "MEDIUM": (0, 255, 255),
        "LOW": (0, 255, 0),
    }
    for z in zones:
        x, y, w, h = z["bbox"]
        color = sev_color[z["severity"]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"#{z['id']} {z['severity']} {z['t_max']:.1f}C"
        cv2.putText(img, label, (x, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return img

def build_report(thermal, zones, threshold_c):
    return {
        "image_stats": {
            "t_min": round(float(thermal.min()), 2),
            "t_max": round(float(thermal.max()), 2),
            "t_mean": round(float(thermal.mean()), 2),
        },
        "threshold_c": threshold_c,
        "zones_count": len(zones),
        "zones": zones,
    }

def main():
    thermal = np.load("sample_thermal.npy")
    threshold_c = 60.0

    zones, mask = find_hot_zones(thermal, threshold_c=threshold_c, min_area=150)
    img = annotate(thermal, zones)
    cv2.imwrite("thermal_annotated.png", img)
    cv2.imwrite("thermal_mask.png", mask)

    report = build_report(thermal, zones, threshold_c)
    with open("report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Zones found: {len(zones)} (threshold {threshold_c}C)")
    for z in zones:
        print(f"  #{z['id']} {z['severity']:8s} t_max={z['t_max']:.1f}C bbox={z['bbox']}")
    print("Saved: thermal_annotated.png, thermal_mask.png, report.json")

if __name__ == "__main__":
    main()
