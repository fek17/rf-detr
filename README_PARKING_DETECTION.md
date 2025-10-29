# 🅿️ Parking Lot Detection for Satellite Images

Detect parking lots and vehicles in satellite/aerial imagery using RF-DETR's segmentation model. Get pixel-perfect masks instead of just bounding boxes!

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies

```bash
# From the rf-detr directory
pip install -e .
```

### 2. Get a Satellite Image

Take a screenshot from Google Maps (satellite view) of a parking lot, or use:
```bash
wget https://media.roboflow.com/notebooks/examples/aerial-parking.jpg -O parking.jpg
```

### 3. Run Detection

```bash
python example_parking_segmentation.py parking.jpg
```

That's it! A window will show your results with pixel-level masks.

## 📸 What You'll Get

Running the example script shows a **4-panel comparison**:

```
┌─────────────────┬─────────────────┐
│ Original Image  │ Bounding Boxes  │
├─────────────────┼─────────────────┤
│ Masks Only      │ Combined View   │
└─────────────────┴─────────────────┘
```

Plus detailed statistics:
- Number of vehicles detected
- Confidence scores
- Pixel-accurate areas
- Coverage percentages
- Parking occupancy estimates

## 🎯 Why Use Segmentation?

**Traditional Detection (Boxes):**
```
┌──────────┐  ← Rectangular box includes
│  ████    │     lots of empty space
│ ██████   │
│  ████    │
└──────────┘
```

**Segmentation (Masks):**
```
  ████       ← Exact shape of the
 ██████         parking lot/vehicle
  ████
```

**Benefits:**
- ✅ Pixel-perfect boundaries for irregular shapes
- ✅ Accurate area calculations
- ✅ Better for identifying parking lot regions
- ✅ Can calculate true occupancy rates

## 📂 What's Included

| File | Purpose |
|------|---------|
| `parking_lot_segmentation_detector.py` | Main segmentation detector (CLI + Python API) |
| `example_parking_segmentation.py` | Interactive demo with visualization |
| `parking_lot_detector.py` | Faster detection-only version (boxes) |
| `example_parking_detection.py` | Simple detection demo |
| `test_installation.py` | Test if everything is working |
| `GETTING_STARTED.md` | Detailed setup guide |
| `PARKING_LOT_DETECTION.md` | Complete documentation |

## 💻 Usage Examples

### Command Line

```bash
# Single image with segmentation
python parking_lot_segmentation_detector.py parking.jpg -o output/

# Batch process a folder
python parking_lot_segmentation_detector.py ./images/ -o output/ --stats

# Extract individual masks
python parking_lot_segmentation_detector.py parking.jpg --extract-masks

# Adjust sensitivity (lower = more detections)
python parking_lot_segmentation_detector.py parking.jpg -t 0.3

# Higher confidence threshold (fewer false positives)
python parking_lot_segmentation_detector.py parking.jpg -t 0.7
```

### Python API

```python
from parking_lot_segmentation_detector import ParkingLotSegmentationDetector
from PIL import Image

# Initialize detector
detector = ParkingLotSegmentationDetector(
    use_segmentation=True,
    threshold=0.5
)

# Detect
image = Image.open("parking.jpg")
detections = detector.detect(image)

print(f"Found {len(detections)} objects")
print(f"Masks shape: {detections.mask.shape}")

# Calculate areas
area_stats = detector.calculate_parking_area(
    detections,
    image_size=image.size
)
print(f"Total parking area: {area_stats['total_parking_area_pixels']:,} pixels")

# Save annotated result
annotated = detector.annotate_image(image, detections, show_masks=True)
Image.fromarray(annotated).save("result.jpg")
```

## 🧪 Test Your Installation

```bash
python test_installation.py
```

This checks:
- Python version
- Dependencies installed
- RF-DETR package working
- Model can initialize
- GPU/CUDA availability

## 📋 Requirements

- Python 3.9+
- PyTorch 1.13+
- Supervision
- PIL/Pillow
- NumPy
- Matplotlib

All installed automatically with `pip install -e .`

## 🎓 What It Detects (Pre-trained Model)

The pre-trained COCO model detects:
- 🚗 Cars
- 🚚 Trucks
- 🚌 Buses
- 🏍️ Motorcycles

**For custom parking lot detection** (lot boundaries, parking spaces, etc.), you'll need to fine-tune on your own dataset. See `PARKING_LOT_DETECTION.md` for details.

## 🔧 Common Options

### Adjust Detection Sensitivity

```bash
# More detections (lower threshold)
-t 0.3

# Fewer false positives (higher threshold)
-t 0.7
```

### Control Mask Visibility

```bash
# More transparent masks
--mask-opacity 0.3

# More opaque masks
--mask-opacity 0.8
```

### Output Options

```bash
# Show statistics
--stats

# Extract individual masks
--extract-masks

# Don't show masks (boxes only)
--no-masks

# Use CPU instead of GPU
--device cpu
```

## 🎯 Typical Workflow

```bash
# 1. Test installation
python test_installation.py

# 2. Try the example with your image
python example_parking_segmentation.py parking_lot.jpg

# 3. Batch process multiple images
python parking_lot_segmentation_detector.py ./my_images/ -o results/ --stats

# 4. Extract masks for analysis
python parking_lot_segmentation_detector.py ./my_images/ -o results/ --extract-masks

# 5. Integrate into your code (see Python API examples)
```

## 📖 Documentation

- **Quick Start:** `GETTING_STARTED.md` - Step-by-step setup
- **Full Guide:** `PARKING_LOT_DETECTION.md` - Complete documentation
- **Fine-tuning:** See the fine-tuning section in `PARKING_LOT_DETECTION.md`

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No detections found" | Lower threshold: `-t 0.3` |
| Out of memory | Use CPU: `--device cpu` |
| Slow inference | Make sure CUDA is available |
| Import errors | Run: `pip install -e .` |

Run `python test_installation.py` to diagnose issues.

## ⚡ Performance

- **First run:** 1-2 minutes (downloads model)
- **With GPU:** 1-5 seconds per image
- **With CPU:** 10-30 seconds per image
- **Model size:** ~200MB

## 🎨 Output Files

After running, check the `output/` folder:

```
output/
├── annotated_parking.jpg          # Image with overlays
├── comparison_parking.jpg         # 4-panel comparison
└── masks/                         # Individual masks
    ├── mask_000_car_0.95.png      # Masked image
    ├── mask_000_car_binary.png    # Binary mask
    └── ...
```

## 🚀 Next Steps

1. **Try it:** Run the example script
2. **Experiment:** Try different thresholds and images
3. **Customize:** Use the Python API in your code
4. **Fine-tune:** Train on your own parking lot dataset for production use

## 📝 Example Output

```
Found 47 objects

Detection Statistics:
  Total detections: 47
  Average confidence: 0.892
  Confidence range: 0.512 - 0.987

Detections by Class:
  car: 42
  truck: 5

Segmentation Statistics:
  Total mask area: 1,234,567 pixels
  Average mask area: 26,267 pixels
  Total coverage: 12.34% of image

Parking Lot Area Analysis:
  Total parking area: 2,456,789 pixels
  Parking coverage: 24.57% of image
```

## 💡 Pro Tips

1. **Better images = better results**
   - Use high-resolution satellite imagery
   - Clear, daytime images work best
   - Overhead view is ideal

2. **Adjust the threshold**
   - Start with 0.5 (default)
   - Lower for more detections
   - Higher for higher precision

3. **Use segmentation for:**
   - Calculating parking lot areas
   - Identifying lot boundaries
   - Accurate occupancy rates

4. **For production use:**
   - Fine-tune on your specific imagery
   - Use batch processing for efficiency
   - Enable optimization (default)

---

**Ready to start?** Run:
```bash
python example_parking_segmentation.py your_parking_image.jpg
```

**Need help?** See `GETTING_STARTED.md` for detailed instructions.
