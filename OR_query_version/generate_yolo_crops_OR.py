import json
import os
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    input_dir = project_root / "vehicle_subset_OR"
    output_dir = project_root / "outputs" / "crops_OR"
    metadata_path = project_root / "outputs" / "crop_metadata_OR.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Small and fast baseline model. You can later swap to yolov8s.pt or a custom model.
    model = YOLO("yolov8n.pt")

    # Keep only the vehicle classes relevant for your project.
    keep_classes = {"car", "bus", "truck", "motorcycle"}

    image_files = sorted(
        [
            p for p in input_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )

    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"Saving crops to {output_dir}")

    all_metadata = []

    for img_idx, image_path in enumerate(image_files, start=1):
        print(f"[{img_idx}/{len(image_files)}] {image_path.name}")

        image = Image.open(image_path).convert("RGB")
        results = model.predict(
            source=str(image_path),
            verbose=False,
            conf=0.25,
            imgsz=640
        )

        per_class_counter: dict[str, int] = {}

        for result in results:
            boxes = result.boxes
            names = result.names

            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls.item())
                cls_name = names[cls_id]

                if cls_name not in keep_classes:
                    continue

                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Clamp to valid image range
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.width, x2)
                y2 = min(image.height, y2)

                # Skip degenerate boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image.crop((x1, y1, x2, y2))

                det_idx = per_class_counter.get(cls_name, 0)
                per_class_counter[cls_name] = det_idx + 1

                crop_filename = f"{image_path.stem}_{cls_name}_{det_idx}{image_path.suffix.lower()}"
                crop_path = output_dir / crop_filename
                crop.save(crop_path)

                all_metadata.append(
                    {
                        "crop_file": crop_filename,
                        "crop_path": str(crop_path),
                        "original_image": image_path.name,
                        "original_path": str(image_path),
                        "class_name": cls_name,
                        "confidence": round(conf, 4),
                        "box_xyxy": [x1, y1, x2, y2],
                        "crop_size": [x2 - x1, y2 - y1],
                    }
                )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print(f"Done. Saved {len(all_metadata)} crops.")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()