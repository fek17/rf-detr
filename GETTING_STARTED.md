# Quick Start Guide: Parking Lot Detection

A simple, step-by-step guide to get started with parking lot detection in satellite images.

## Prerequisites

You need Python 3.9+ installed. This code works on Linux, macOS, and Windows.

## Step 1: Install RF-DETR

If you haven't already, install the RF-DETR package:

```bash
# Install from the repository
pip install -e .

# Or if you cloned the repo, install dependencies
pip install torch torchvision supervision pillow numpy matplotlib
```

**Note:** The first time you run the code, it will automatically download the model weights (~200MB for segmentation model). This only happens once.

## Step 2: Get a Satellite Image

You need a satellite/aerial image of a parking lot. Here are some easy ways to get one:

### Option A: Google Earth Pro (Free, Desktop App)
1. Download [Google Earth Pro](https://www.google.com/earth/versions/#earth-pro) (it's free!)
2. Navigate to a parking lot
3. File → Save → Save Image
4. Save as a JPG or PNG file

### Option B: Screenshot from Online Maps
1. Go to [Google Maps](https://maps.google.com) or [Bing Maps](https://bing.com/maps)
2. Switch to Satellite view
3. Find a parking lot (malls, airports, office buildings work great)
4. Take a screenshot and save as JPG/PNG

### Option C: Use a Test Image
Download a sample parking lot image:
```bash
# Example parking lot image
wget https://media.roboflow.com/notebooks/examples/aerial-parking.jpg -O parking_lot.jpg

# Or use curl
curl -o parking_lot.jpg https://media.roboflow.com/notebooks/examples/aerial-parking.jpg
```

## Step 3: Run Detection

### Quick Test (Simplest Method)

```bash
# Navigate to the rf-detr directory
cd /path/to/rf-detr

# Run the example script with your image
python example_parking_segmentation.py parking_lot.jpg
```

**What happens:**
- The model downloads automatically (first time only)
- Detects parking lots/vehicles with pixel-level masks
- Shows a 4-panel comparison visualization
- Prints detailed statistics
- Saves results to `output/` folder

### Batch Processing

Process multiple images at once:

```bash
# Process all images in a folder
python parking_lot_segmentation_detector.py ./my_images/ -o output/

# With statistics
python parking_lot_segmentation_detector.py ./my_images/ -o output/ --stats

# Extract individual masks for each detection
python parking_lot_segmentation_detector.py ./my_images/ -o output/ --extract-masks
```

### Single Image with Options

```bash
# Basic detection
python parking_lot_segmentation_detector.py parking_lot.jpg -o output/

# Lower threshold to detect more objects (more sensitive)
python parking_lot_segmentation_detector.py parking_lot.jpg -o output/ -t 0.3

# Higher threshold for fewer false positives (more strict)
python parking_lot_segmentation_detector.py parking_lot.jpg -o output/ -t 0.7

# Show statistics
python parking_lot_segmentation_detector.py parking_lot.jpg -o output/ --stats

# Extract individual masks
python parking_lot_segmentation_detector.py parking_lot.jpg -o output/ --extract-masks

# Adjust mask transparency (0.0 = transparent, 1.0 = opaque)
python parking_lot_segmentation_detector.py parking_lot.jpg -o output/ --mask-opacity 0.7
```

## Step 4: Check Your Results

After running, check the `output/` folder:

```
output/
├── annotated_parking_lot.jpg    # Image with detections overlaid
├── comparison_parking_lot.jpg   # 4-panel comparison (if using example script)
└── masks/                       # Individual mask images (if --extract-masks used)
    ├── mask_000_car_0.95.png
    ├── mask_000_car_binary.png
    └── ...
```

## Step 5: Use in Python Code

For more control, use the Python API:

```python
from parking_lot_segmentation_detector import ParkingLotSegmentationDetector
from PIL import Image

# Initialize detector (model downloads automatically first time)
detector = ParkingLotSegmentationDetector(
    use_segmentation=True,     # Use segmentation model
    threshold=0.5,              # Confidence threshold
)

# Load your image
image = Image.open("parking_lot.jpg")

# Detect parking lots/vehicles
detections = detector.detect(image)

# See what was found
print(f"Found {len(detections)} objects")

# Get statistics
stats = detector.get_statistics(detections)
print(f"Average confidence: {stats['avg_confidence']:.2f}")

# Save annotated image
annotated = detector.annotate_image(image, detections, show_masks=True)
Image.fromarray(annotated).save("result.jpg")

print("Saved to result.jpg")
```

## Common Issues & Solutions

### Issue: "No module named 'rfdetr'"

**Solution:** Install the package first:
```bash
cd /path/to/rf-detr
pip install -e .
```

### Issue: "No detections found"

**Solutions:**
1. Lower the threshold: `python ... -t 0.3`
2. Make sure your image shows parking lots/vehicles from above
3. Check image quality (should be clear, not too blurry)

### Issue: Out of memory / GPU errors

**Solutions:**
1. Use CPU instead: `python ... --device cpu`
2. Process one image at a time instead of batches
3. Use a smaller image (resize to 1024x1024 or smaller)

### Issue: Slow inference

**Solutions:**
1. Make sure you have GPU available (CUDA)
2. Model optimization is enabled by default (should be fast)
3. First inference is always slower (model compilation)

### Issue: "Model downloading failed"

**Solution:** Check internet connection, or manually download:
```bash
# The model will be saved to current directory
wget https://storage.googleapis.com/rfdetr/rf-detr-seg-preview.pt
```

## Expected Performance

- **First run:** 1-2 minutes (downloading model)
- **Subsequent runs:** 1-5 seconds per image (with GPU)
- **CPU only:** 10-30 seconds per image
- **Model size:** ~200MB for segmentation model

## What to Expect

### With Pre-trained COCO Model:
- ✅ Detects: cars, trucks, buses, motorcycles
- ✅ Works on: parking lots, streets, satellite imagery
- ❌ Won't detect: parking lot boundaries, empty parking spaces

### For Better Results:
**Fine-tune on custom dataset** to detect:
- Parking lot regions/boundaries
- Individual parking spaces
- Occupied vs. empty spaces
- Different parking lot types

See `PARKING_LOT_DETECTION.md` for fine-tuning instructions.

## Quick Comparison: Which Script to Use?

| Script | Use Case | Speed | Features |
|--------|----------|-------|----------|
| `example_parking_segmentation.py` | Learning/Demo | Medium | Interactive, 4-panel viz, stats |
| `parking_lot_segmentation_detector.py` | Production | Fast | Batch processing, CLI, customizable |
| `parking_lot_detector.py` | Boxes only | Fastest | Bounding boxes, no masks |

**Recommendation:** Start with `example_parking_segmentation.py` to see what's possible, then use `parking_lot_segmentation_detector.py` for real work.

## Complete Example Workflow

```bash
# 1. Get a satellite image (screenshot from Google Maps)
# Save it as parking_lot.jpg

# 2. Run detection
python example_parking_segmentation.py parking_lot.jpg

# 3. View results
# - Window pops up showing 4-panel comparison
# - Results saved to output/segmented_parking_lot.jpg
# - Statistics printed in terminal

# 4. For batch processing
mkdir my_parking_images
# Add your images to my_parking_images/

python parking_lot_segmentation_detector.py my_parking_images/ \
    -o results/ \
    --stats \
    --extract-masks \
    -t 0.5

# 5. Check results folder
ls results/
# annotated_image1.jpg, annotated_image2.jpg, ...
# masks/image1/, masks/image2/, ...
```

## Next Steps

1. **Try it:** Run the example script with a satellite image
2. **Experiment:** Adjust threshold, try different images
3. **Fine-tune:** For production use, fine-tune on your specific parking lot dataset
4. **Integrate:** Use the Python API to integrate into your application

## Need Help?

- See detailed examples: `PARKING_LOT_DETECTION.md`
- RF-DETR docs: Check the `docs/` folder
- Issues: Check the [GitHub issues](https://github.com/roboflow/rf-detr/issues)

Happy detecting! 🚗🅿️
