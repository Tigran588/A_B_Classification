# yolo_box_checker.py
# Works for: (A) single image+label  (B) entire folders
# Install: pip install opencv-python pyyaml

import os, cv2, glob, argparse, re

def parse_label_file(lbl_path):
    boxes, cls = [], []
    if not os.path.exists(lbl_path):
        return boxes, cls, "missing_label"
    with open(lbl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                return [], [], f"bad_line_{i}"
            try:
                cid = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
            except Exception:
                return [], [], f"parse_error_line_{i}"
            boxes.append([x, y, w, h]); cls.append(cid)
    return boxes, cls, None

def norm_to_xyxy(b, W, H):
    x, y, w, h = b
    xc, yc = x * W, y * H
    bw, bh = w * W, h * H
    return [int(xc - bw/2), int(yc - bh/2), int(xc + bw/2), int(yc + bh/2)]

def check_and_draw(img_path, lbl_path, out_dir, expected_h=None, expected_w=None, allowed_classes=None):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(img_path)
    if img is None:
        return [f"cannot_read_image:{img_path}"]
    H, W = img.shape[:2]
    issues = []

    if expected_h and expected_w and (H, W) != (expected_h, expected_w):
        issues.append(f"wrong_size {W}x{H} (expected {expected_w}x{expected_h})")

    boxes, cls, err = parse_label_file(lbl_path)
    if err:
        issues.append(err)
    else:
        for i, (b, c) in enumerate(zip(boxes, cls), 1):
            if allowed_classes is not None and c not in allowed_classes:
                issues.append(f"class_id_out_of_range line {i}: {c}")
            x, y, w, h = b
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                issues.append(f"norm_out_of_range line {i}: {x:.3f},{y:.3f},{w:.3f},{h:.3f}")
            x1, y1, x2, y2 = norm_to_xyxy(b, W, H)
            if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                issues.append(f"box_outside_image line {i}: xyxy={x1},{y1},{x2},{y2}")
            cv2.rectangle(img, (max(0,x1), max(0,y1)), (min(W-1,x2), min(H-1,y2)), (255,200,0), 2)
            cv2.putText(img, str(c), (max(0,x1), max(12,y1-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)

    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(img_path))[0] + "_vis.jpg")
    cv2.imwrite(out_path, img)
    return issues

def iter_image_files(images_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    for p in sorted(glob.glob(os.path.join(images_dir, "*"))):
        if p.lower().endswith(exts):
            yield p

def main():
    ap = argparse.ArgumentParser(description="YOLO label checker & visualizer")
    ap.add_argument("--img", type=str, help="single image path")
    ap.add_argument("--lbl", type=str, help="single label path")
    ap.add_argument("--images_dir", type=str, help="folder with images")
    ap.add_argument("--labels_dir", type=str, help="folder with labels (same stems as images)")
    ap.add_argument("--out", type=str, default="./_debug_boxes", help="output folder for visualizations")
    ap.add_argument("--expected_h", type=int, default=None, help="expected image height (optional)")
    ap.add_argument("--expected_w", type=int, default=None, help="expected image width (optional)")
    ap.add_argument("--classes", type=str, default=None, help="allowed class ids, e.g. '0,1'")
    args = ap.parse_args()

    allowed = None
    if args.classes:
        try:
            allowed = set(int(x) for x in re.split(r"[,;\s]+", args.classes.strip()) if x != "")
        except Exception:
            print("[WARN] --classes could not be parsed; ignoring.")

    problems = []

    # Single file mode
    if args.img and args.lbl:
        issues = check_and_draw(args.img, args.lbl, args.out, args.expected_h, args.expected_w, allowed)
        if issues:
            problems.append((args.img, ";".join(issues)))
    # Folder mode
    elif args.images_dir and args.labels_dir:
        for ip in iter_image_files(args.images_dir):
            stem = os.path.splitext(os.path.basename(ip))[0]
            lp = os.path.join(args.labels_dir, stem + ".txt")
            if not os.path.exists(lp):
                problems.append((ip, "missing_label"))
                # still draw image without boxes
                check_and_draw(ip, lp, args.out, args.expected_h, args.expected_w, allowed)
                continue
            issues = check_and_draw(ip, lp, args.out, args.expected_h, args.expected_w, allowed)
            if issues:
                problems.append((ip, ";".join(issues)))
    else:
        print("Provide either --img & --lbl  OR  --images_dir & --labels_dir")
        return

    print("\n==== SUMMARY ====")
    print(f"Visualizations saved to: {os.path.abspath(args.out)}")
    if problems:
        print(f"Problems: {len(problems)}")
        for p, why in problems[:200]:
            print(f"- {p} -> {why}")
        if len(problems) > 200:
            print(f"... and {len(problems)-200} more")
    else:
        print("No issues detected.")

if __name__ == "__main__":
    main()
