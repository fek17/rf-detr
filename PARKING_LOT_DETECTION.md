# Parking Lot Detection in Satellite Images

This guide explains how to use RF-DETR to detect parking lots and vehicles in satellite imagery.

## 🎯 Recommended Approach: Use Segmentation!

**For the most accurate parking lot detection, use the segmentation model** (`parking_lot_segmentation_detector.py`). Segmentation provides pixel-level masks instead of just bounding boxes, which is much better for identifying parking lot regions and boundaries.

### Why Segmentation is Better:
- ✅ **Pixel-perfect boundaries** - Get exact parking lot outlines, not just boxes
- ✅ **Accurate area calculations** - Calculate precise parking lot sizes
- ✅ **Better for regions** - Identify large parking lot areas, not just individual cars
- ✅ **Mask overlap analysis** - Understand parking space usage better
- ✅ **Works great in Roboflow Playground** - As you've experienced!

### Quick Start with Segmentation:

```bash
# Basic usage - detects with pixel-level masks
python parking_lot_segmentation_detector.py satellite_image.jpg -o output/

# Try the interactive example
python example_parking_segmentation.py satellite_image.jpg

# With statistics and mask extraction
python parking_lot_segmentation_detector.py image.jpg -o output/ --stats --extract-masks
```

See the [Segmentation Section](#segmentation-based-detection) below for detailed usage.

---

## Overview

We provide two approaches for parking lot detection:

1. **Segmentation-based** (`parking_lot_segmentation_detector.py`) - **RECOMMENDED**
   - Uses RF-DETR-Seg model with pixel-level masks
   - Better for identifying parking lot regions and boundaries
   - Accurate area calculations

2. **Detection-based** (`parking_lot_detector.py`) - Faster, simpler
   - Uses bounding boxes only
   - Good for counting vehicles
   - Faster inference

### Detection-Only Features

The `parking_lot_detector.py` script provides an easy-to-use interface for detecting parking lots in satellite images. It supports:

- Single image or batch processing
- Multiple model sizes (small, medium, base)
- Custom fine-tuned models
- Class filtering (e.g., only detect cars, trucks, buses)
- Confidence thresholding
- Statistics and visualization

## Quick Start

### 1. Basic Usage with Pre-trained Model

The pre-trained COCO model can detect vehicles (cars, trucks, buses), which is useful for identifying parking areas:

```bash
# Detect vehicles in a single satellite image
python parking_lot_detector.py satellite_image.jpg -o output/

# Process all images in a directory
python parking_lot_detector.py satellite_images/ -o output/

# Use smaller/faster model
python parking_lot_detector.py image.jpg -m small -o output/

# Adjust confidence threshold
python parking_lot_detector.py image.jpg -t 0.3 -o output/

# Show detection statistics
python parking_lot_detector.py image.jpg -o output/ --stats
```

### 2. Python API Usage

```python
from parking_lot_detector import ParkingLotDetector
from PIL import Image
import matplotlib.pyplot as plt

# Initialize detector
detector = ParkingLotDetector(
    model_size="medium",  # or "small", "base"
    threshold=0.5,
    target_classes=["car", "truck", "bus"]  # Filter specific classes
)

# Detect vehicles in image
image = Image.open("satellite_parking.jpg")
detections = detector.detect(image)

print(f"Found {len(detections)} vehicles")
print(f"Classes detected: {detections.class_id}")
print(f"Confidences: {detections.confidence}")
print(f"Bounding boxes: {detections.xyxy}")

# Annotate and visualize
annotated = detector.annotate_image(image, detections)
plt.figure(figsize=(15, 10))
plt.imshow(annotated)
plt.axis('off')
plt.show()

# Get statistics
stats = detector.get_statistics(detections)
print(f"Total detections: {stats['total_detections']}")
print(f"Average confidence: {stats['avg_confidence']:.2f}")
print(f"Detections by class: {stats['detections_by_class']}")
```

### 3. Batch Processing

```python
from parking_lot_detector import ParkingLotDetector
from pathlib import Path

detector = ParkingLotDetector(model_size="medium")

# Get all satellite images
image_paths = list(Path("satellite_images/").glob("*.jpg"))

# Process in batch (more efficient)
detections_list = detector.detect_batch(image_paths)

# Save results
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

for img_path, detections in zip(image_paths, detections_list):
    # Load original image
    image = Image.open(img_path)

    # Annotate
    annotated = detector.annotate_image(image, detections)

    # Save
    output_path = output_dir / f"annotated_{img_path.name}"
    Image.fromarray(annotated).save(output_path)

    print(f"{img_path.name}: {len(detections)} vehicles detected")
```

## Segmentation-Based Detection

The segmentation model (RF-DETR-Seg-Preview) provides pixel-level masks for much more accurate parking lot detection compared to bounding boxes alone.

### Why Use Segmentation?

**Problem with bounding boxes:**  Rectangular boxes don't capture the actual shape of parking lots, which can be L-shaped, curved, or irregular. You get lots of "empty" space in the box that isn't actually part of the parking lot.

**Solution with segmentation:** Pixel-perfect masks that follow the exact outline of parking lots, vehicles, and parking spaces.

### Basic Usage

```python
from parking_lot_segmentation_detector import ParkingLotSegmentationDetector
from PIL import Image

# Initialize with segmentation model
detector = ParkingLotSegmentationDetector(
    use_segmentation=True,  # Enable segmentation
    model_size="seg-preview",  # Segmentation model
    threshold=0.5
)

# Detect with masks
image = Image.open("satellite_parking.jpg")
detections = detector.detect(image)

# Detections now include pixel-level masks
print(f"Found {len(detections)} objects")
print(f"Masks shape: {detections.mask.shape}")  # (N, H, W) - one mask per detection

# Visualize with masks
annotated = detector.annotate_image(
    image,
    detections,
    show_masks=True,  # Show segmentation masks
    mask_opacity=0.5
)
```

### Command-Line Usage

```bash
# Basic segmentation
python parking_lot_segmentation_detector.py satellite_image.jpg -o output/

# With statistics
python parking_lot_segmentation_detector.py image.jpg -o output/ --stats

# Extract individual masks as separate images
python parking_lot_segmentation_detector.py image.jpg -o output/ --extract-masks

# Adjust mask opacity
python parking_lot_segmentation_detector.py image.jpg -o output/ --mask-opacity 0.7

# Process directory
python parking_lot_segmentation_detector.py images/ -o output/ --stats --extract-masks
```

### Calculate Parking Lot Areas

```python
from parking_lot_segmentation_detector import ParkingLotSegmentationDetector
from PIL import Image

detector = ParkingLotSegmentationDetector(use_segmentation=True)

image = Image.open("satellite_parking.jpg")
detections = detector.detect(image)

# Calculate area statistics
area_stats = detector.calculate_parking_area(
    detections,
    image_size=image.size,
    pixel_to_meter_ratio=0.5  # Optional: 0.5 meters per pixel
)

print(f"Total parking area: {area_stats['total_parking_area_pixels']:,} pixels")
print(f"Coverage: {area_stats['parking_coverage_percentage']:.1f}% of image")

if 'total_parking_area_sq_meters' in area_stats:
    print(f"Area: {area_stats['total_parking_area_sq_meters']:.0f} square meters")
```

### Extract and Analyze Individual Masks

```python
from parking_lot_segmentation_detector import ParkingLotSegmentationDetector
from PIL import Image
import numpy as np

detector = ParkingLotSegmentationDetector(use_segmentation=True)

image = Image.open("satellite_parking.jpg")
detections = detector.detect(image)

# Extract masks as separate images
mask_paths = detector.extract_masks_as_images(
    image,
    detections,
    output_dir="output/masks"
)
print(f"Saved {len(mask_paths)} mask images")

# Analyze each mask
for i, mask in enumerate(detections.mask):
    area = np.sum(mask)  # Number of pixels in mask
    bbox = detections.xyxy[i]
    class_id = detections.class_id[i]

    # Calculate how much of bounding box is actual object
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    fill_ratio = area / bbox_area

    print(f"Object {i}: {COCO_CLASSES[class_id]}")
    print(f"  Mask area: {area:,} pixels")
    print(f"  BBox area: {bbox_area:.0f} pixels")
    print(f"  Fill ratio: {fill_ratio:.1%}")
```

### Comparison Visualization

The example script shows a 4-panel comparison:

```bash
python example_parking_segmentation.py satellite_image.jpg
```

This creates:
1. Original image
2. Bounding boxes only
3. Segmentation masks only
4. Combined (boxes + masks)

### Advanced: Combine Multiple Masks

```python
import numpy as np
from parking_lot_segmentation_detector import ParkingLotSegmentationDetector
from PIL import Image

detector = ParkingLotSegmentationDetector(use_segmentation=True)

image = Image.open("satellite_parking.jpg")
detections = detector.detect(image)

# Create combined mask of all parking lot regions
combined_mask = np.zeros(detections.mask[0].shape, dtype=bool)
for mask in detections.mask:
    combined_mask = combined_mask | mask  # Union of all masks

# Calculate total parking lot area
total_area = np.sum(combined_mask)
print(f"Total parking lot area: {total_area:,} pixels")

# Visualize combined mask
combined_mask_image = np.zeros((*combined_mask.shape, 3), dtype=np.uint8)
combined_mask_image[combined_mask] = [255, 0, 0]  # Red for parking areas

# Overlay on original
import cv2
overlay = cv2.addWeighted(np.array(image), 0.7, combined_mask_image, 0.3, 0)
Image.fromarray(overlay).save("parking_lot_overlay.jpg")
```

### Segmentation Statistics

```python
detector = ParkingLotSegmentationDetector(use_segmentation=True)
detections = detector.detect("satellite_parking.jpg")

stats = detector.get_statistics(detections)

# Regular detection stats
print(f"Total detections: {stats['total_detections']}")
print(f"Average confidence: {stats['avg_confidence']:.2f}")

# Segmentation-specific stats
if 'segmentation' in stats:
    seg = stats['segmentation']
    print(f"\nSegmentation Stats:")
    print(f"  Total mask area: {seg['total_mask_area']:,} pixels")
    print(f"  Average mask area: {seg['avg_mask_area']:.0f} pixels")
    print(f"  Coverage: {seg['total_coverage_percentage']:.2f}%")

    # Individual mask areas
    for i, area in enumerate(seg['mask_areas']):
        print(f"  Object {i}: {area:,} pixels")
```

### Tips for Segmentation

1. **Better than boxes for:**
   - Irregular parking lot shapes
   - Calculating accurate areas
   - Identifying exact boundaries
   - Analyzing parking space layout

2. **Visualization tips:**
   - Use `mask_opacity=0.5` for balanced visibility
   - Set `mask_opacity=0.7` to emphasize masks
   - Use `show_masks=False` to see just bounding boxes

3. **Performance:**
   - Segmentation is slightly slower than detection-only
   - Still real-time capable (~10-30 FPS depending on model)
   - Enable optimization for best performance

4. **Fine-tuning:**
   - Same fine-tuning process as detection model
   - Add segmentation annotations to your dataset
   - Model learns both boxes and pixel-level masks

## Fine-tuning for Parking Lot Detection

For better results, fine-tune the model on a parking lot dataset. This allows the model to:
- Detect parking lot boundaries (not just vehicles)
- Identify parking spaces (occupied/empty)
- Recognize parking lot types (surface, multi-level, etc.)

### Step 1: Prepare Your Dataset

Create a COCO-format dataset with annotations for parking lots:

```
parking_lot_dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── satellite_1.jpg
│   ├── satellite_2.jpg
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   └── ...
└── test/
    ├── _annotations.coco.json
    └── ...
```

**Annotation format** (`_annotations.coco.json`):
```json
{
  "categories": [
    {"id": 0, "name": "parking_lot"},
    {"id": 1, "name": "parking_space"},
    {"id": 2, "name": "car"},
    {"id": 3, "name": "truck"}
  ],
  "images": [
    {
      "id": 1,
      "file_name": "satellite_1.jpg",
      "width": 1024,
      "height": 1024
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [100, 100, 300, 200],
      "area": 60000,
      "iscrowd": 0
    }
  ]
}
```

**Tools for creating annotations:**
- [Roboflow](https://roboflow.com/) - Easy annotation and export to COCO format
- [CVAT](https://github.com/opencv/cvat) - Open-source annotation tool
- [LabelMe](https://github.com/wkentaro/labelme) - Python annotation tool
- [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/) - Web-based tool

### Step 2: Fine-tune the Model

```python
from rfdetr import RFDETRMedium

# Load pre-trained model
model = RFDETRMedium()

# Fine-tune on parking lot dataset
model.train(
    dataset_dir="parking_lot_dataset/",
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,  # Effective batch size = 16
    lr=1e-4,
    lr_encoder=1.5e-4,
    output_dir="parking_model_output/",
    device="cuda",
    use_ema=True,
    tensorboard=True,  # Enable TensorBoard logging
    multi_scale=True,  # Multi-scale training for better generalization
)
```

**Training parameters guide:**
- `epochs=100`: More epochs for better convergence (monitor validation loss)
- `batch_size=4`: Adjust based on GPU memory (4 for 8GB, 8 for 16GB+)
- `grad_accum_steps=4`: Accumulate gradients for larger effective batch size
- `lr=1e-4`: Learning rate for decoder
- `lr_encoder=1.5e-4`: Higher LR for encoder (fine-tuning)
- `use_ema=True`: Exponential Moving Average for better final model
- `multi_scale=True`: Train on multiple image scales for robustness

### Step 3: Use Fine-tuned Model

```python
from parking_lot_detector import ParkingLotDetector

# Load fine-tuned model
detector = ParkingLotDetector(
    model_size="medium",
    pretrain_weights="parking_model_output/checkpoint_best_regular.pth",
    threshold=0.5,
    target_classes=["parking_lot", "parking_space", "car"]  # Your custom classes
)

# Use as before
detections = detector.detect("satellite_image.jpg")
```

**Command-line usage:**
```bash
python parking_lot_detector.py satellite_image.jpg \
    -w parking_model_output/checkpoint_best_regular.pth \
    -c parking_lot parking_space car \
    -t 0.5 \
    -o output/ \
    --stats
```

## Advanced Examples

### 1. Counting Parking Spaces

```python
from parking_lot_detector import ParkingLotDetector
import numpy as np

detector = ParkingLotDetector(
    model_size="medium",
    target_classes=["parking_space"]
)

image_path = "parking_lot_satellite.jpg"
detections = detector.detect(image_path)

# Count total parking spaces
total_spaces = len(detections)

# Separate occupied vs. empty (requires fine-tuned model with occupancy labels)
# Assuming class_id 0 = empty, 1 = occupied
empty_spaces = np.sum(detections.class_id == 0)
occupied_spaces = np.sum(detections.class_id == 1)

print(f"Total parking spaces: {total_spaces}")
print(f"Empty: {empty_spaces}, Occupied: {occupied_spaces}")
print(f"Occupancy rate: {occupied_spaces / total_spaces * 100:.1f}%")
```

### 2. Processing Large Satellite Images

For very large satellite images, process them in tiles:

```python
from parking_lot_detector import ParkingLotDetector
from PIL import Image
import numpy as np

def tile_image(image, tile_size=1024, overlap=128):
    """Split large image into overlapping tiles."""
    width, height = image.size
    tiles = []
    positions = []

    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)

            tile = image.crop((x, y, x_end, y_end))
            tiles.append(tile)
            positions.append((x, y))

    return tiles, positions

def merge_detections(detections_list, positions):
    """Merge detections from tiles back to original coordinates."""
    all_boxes = []
    all_confidences = []
    all_class_ids = []

    for detections, (x_offset, y_offset) in zip(detections_list, positions):
        if len(detections) > 0:
            # Adjust box coordinates
            boxes = detections.xyxy.copy()
            boxes[:, [0, 2]] += x_offset
            boxes[:, [1, 3]] += y_offset

            all_boxes.append(boxes)
            all_confidences.append(detections.confidence)
            all_class_ids.append(detections.class_id)

    if not all_boxes:
        return sv.Detections.empty()

    # Concatenate all detections
    merged_detections = sv.Detections(
        xyxy=np.vstack(all_boxes),
        confidence=np.concatenate(all_confidences),
        class_id=np.concatenate(all_class_ids)
    )

    # Apply Non-Maximum Suppression to remove duplicates
    merged_detections = merged_detections.with_nms(threshold=0.5)

    return merged_detections

# Process large satellite image
detector = ParkingLotDetector(model_size="medium")

large_image = Image.open("large_satellite_image.tif")
print(f"Image size: {large_image.size}")

# Split into tiles
tiles, positions = tile_image(large_image, tile_size=1024, overlap=128)
print(f"Split into {len(tiles)} tiles")

# Detect in each tile
detections_list = detector.detect_batch(tiles)

# Merge results
merged_detections = merge_detections(detections_list, positions)
print(f"Total detections: {len(merged_detections)}")

# Annotate original image
annotated = detector.annotate_image(large_image, merged_detections)
Image.fromarray(annotated).save("large_annotated.jpg")
```

### 3. Export Detection Results

```python
from parking_lot_detector import ParkingLotDetector
import json
import pandas as pd

detector = ParkingLotDetector(model_size="medium")
detections = detector.detect("satellite_image.jpg")

# Export to JSON
results = {
    "image": "satellite_image.jpg",
    "detections": [
        {
            "class": COCO_CLASSES[class_id],
            "confidence": float(confidence),
            "bbox": [float(x) for x in bbox]
        }
        for class_id, confidence, bbox in zip(
            detections.class_id,
            detections.confidence,
            detections.xyxy
        )
    ]
}

with open("detections.json", "w") as f:
    json.dump(results, f, indent=2)

# Export to CSV
df = pd.DataFrame({
    "class": [COCO_CLASSES[cid] for cid in detections.class_id],
    "confidence": detections.confidence,
    "x1": detections.xyxy[:, 0],
    "y1": detections.xyxy[:, 1],
    "x2": detections.xyxy[:, 2],
    "y2": detections.xyxy[:, 3],
})
df.to_csv("detections.csv", index=False)

# Export to GeoJSON (if you have georeferencing info)
# This requires rasterio for reading geospatial metadata
import rasterio
from shapely.geometry import box

with rasterio.open("satellite_image.tif") as src:
    transform = src.transform
    crs = src.crs

    features = []
    for class_id, confidence, bbox in zip(
        detections.class_id,
        detections.confidence,
        detections.xyxy
    ):
        # Convert pixel coordinates to geographic coordinates
        x1, y1 = rasterio.transform.xy(transform, bbox[1], bbox[0])
        x2, y2 = rasterio.transform.xy(transform, bbox[3], bbox[2])

        feature = {
            "type": "Feature",
            "geometry": box(x1, y1, x2, y2).__geo_interface__,
            "properties": {
                "class": COCO_CLASSES[class_id],
                "confidence": float(confidence)
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": str(crs)}},
        "features": features
    }

    with open("detections.geojson", "w") as f:
        json.dump(geojson, f, indent=2)
```

## Command-Line Options

```
usage: parking_lot_detector.py [-h] [-o OUTPUT] [-m {small,medium,base}]
                               [-w WEIGHTS] [-t THRESHOLD]
                               [-c CLASSES [CLASSES ...]] [--no-optimize]
                               [--no-labels] [--no-confidence] [--device DEVICE]
                               [--stats]
                               input

positional arguments:
  input                 Path to input image or directory of images

optional arguments:
  -h, --help            Show help message
  -o, --output          Output directory (default: output)
  -m, --model           Model size: small, medium, base (default: medium)
  -w, --weights         Path to custom weights for fine-tuned model
  -t, --threshold       Confidence threshold 0-1 (default: 0.5)
  -c, --classes         Classes to detect (default: car truck bus)
  --no-optimize         Disable model optimization (slower but may use less memory)
  --no-labels           Don't show class labels on output
  --no-confidence       Don't show confidence scores on output
  --device              Device to use: cuda or cpu (default: auto)
  --stats               Print detection statistics
```

## Tips for Best Results

1. **Model Selection:**
   - Use `small` for real-time processing (faster but less accurate)
   - Use `medium` for best balance (recommended)
   - Use `base` or `large` for maximum accuracy

2. **Threshold Tuning:**
   - Lower threshold (0.3-0.4) for detecting small/distant vehicles
   - Higher threshold (0.6-0.7) for fewer false positives
   - Use `--stats` to see confidence distributions

3. **Fine-tuning:**
   - Collect diverse satellite images (different times, seasons, locations)
   - Annotate at least 500-1000 images for good results
   - Use data augmentation during training (built-in with `multi_scale=True`)
   - Monitor validation loss to avoid overfitting

4. **Image Quality:**
   - Higher resolution images give better detection
   - Ensure good contrast and lighting
   - RGB images work best (convert if grayscale)

5. **GPU Usage:**
   - Use CUDA for 10-50x speedup
   - Enable optimization with `optimize=True` (default)
   - Batch process multiple images for efficiency

## Example Datasets

Public datasets for parking lot detection:

1. **PKLot Dataset**: Brazilian parking lot images with occupancy labels
2. **CNRPark-EXT**: Outdoor parking lot dataset
3. **CARPK**: Overhead parking lot images with car counts
4. **Parking Lot Database**: Various parking facilities

You can also create your own dataset using satellite imagery from:
- Google Earth
- Satellite APIs (Planet, Sentinel, Landsat)
- Drone imagery
- Commercial satellite providers

## Troubleshooting

**Issue: Out of memory errors**
- Use smaller model (`-m small`)
- Reduce batch size in Python API
- Process images individually instead of batches
- Disable optimization (`--no-optimize`)

**Issue: Low detection accuracy**
- Lower confidence threshold (`-t 0.3`)
- Fine-tune on your specific satellite imagery
- Ensure images are RGB and good quality
- Check that your target classes are in COCO or custom model

**Issue: Slow inference**
- Ensure CUDA is available: `torch.cuda.is_available()`
- Enable optimization (default, or explicitly set `optimize=True`)
- Use smaller model
- Process in batches

**Issue: Too many false positives**
- Increase confidence threshold (`-t 0.7`)
- Filter specific classes (`-c car truck`)
- Fine-tune on your data for better precision

## Next Steps

1. Try the basic example with pre-trained model
2. Annotate your own parking lot dataset
3. Fine-tune the model on your data
4. Integrate into your application or pipeline
5. Experiment with different model sizes and thresholds

For more information on RF-DETR, see the main [README.md](README.md) and [documentation](docs/).
