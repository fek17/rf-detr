#!/usr/bin/env python3
"""
Simple example script for parking lot detection in satellite images.

This script demonstrates how to use the ParkingLotDetector class to:
1. Load a satellite image
2. Detect vehicles (cars, trucks, buses)
3. Visualize and save the results
4. Print detection statistics

Usage:
    python example_parking_detection.py path/to/satellite_image.jpg
"""

import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import our parking lot detector
from parking_lot_detector import ParkingLotDetector


def main():
    parser = argparse.ArgumentParser(
        description="Simple example for parking lot detection in satellite images"
    )
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to satellite image (optional - creates demo if not provided)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        choices=["small", "medium", "base"],
        default="medium",
        help="Model size (default: medium)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display the image (just save it)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RF-DETR Parking Lot Detection Example")
    print("=" * 60)

    # Initialize the detector
    print(f"\n1. Initializing detector with {args.model.upper()} model...")
    detector = ParkingLotDetector(
        model_size=args.model,
        threshold=args.threshold,
        target_classes=["car", "truck", "bus"],  # Focus on vehicles
        optimize=True  # Enable optimization for faster inference
    )
    print("   Model loaded successfully!")

    # Load image
    if args.image_path is None:
        print("\n2. No image provided. To use this script:")
        print("   python example_parking_detection.py path/to/satellite_image.jpg")
        print("\n   You can download satellite images from:")
        print("   - Google Earth")
        print("   - Bing Maps")
        print("   - OpenStreetMap")
        print("   - Free satellite image datasets (DOTA, DIOR, etc.)")
        return

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"\nError: Image not found at {image_path}")
        return

    print(f"\n2. Loading image from {image_path}...")
    image = Image.open(image_path).convert("RGB")
    print(f"   Image size: {image.size[0]}x{image.size[1]} pixels")

    # Run detection
    print(f"\n3. Running detection (threshold={args.threshold})...")
    detections = detector.detect(image, threshold=args.threshold)
    print(f"   Found {len(detections)} vehicles!")

    # Get statistics
    if len(detections) > 0:
        stats = detector.get_statistics(detections)

        print("\n4. Detection Statistics:")
        print(f"   Total detections: {stats['total_detections']}")
        print(f"   Average confidence: {stats['avg_confidence']:.3f}")
        print(f"   Confidence range: {stats['min_confidence']:.3f} - {stats['max_confidence']:.3f}")

        if stats['detections_by_class']:
            print(f"   Detections by class:")
            for class_name, count in stats['detections_by_class'].items():
                print(f"     - {class_name}: {count}")

        # Print individual detections
        print("\n   Individual detections:")
        from rfdetr.util.coco_classes import COCO_CLASSES
        for i, (class_id, conf, bbox) in enumerate(
            zip(detections.class_id, detections.confidence, detections.xyxy)
        ):
            class_name = COCO_CLASSES[class_id]
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            print(f"     {i+1}. {class_name} (conf={conf:.3f}) at ({x1:.0f},{y1:.0f}) size {w:.0f}x{h:.0f}")
    else:
        print("\n4. No detections found!")
        print("   Try lowering the threshold: -t 0.3")

    # Annotate image
    print("\n5. Annotating image...")
    annotated = detector.annotate_image(
        image,
        detections,
        show_labels=True,
        show_confidence=True
    )

    # Save result
    output_path = Path("output") / f"annotated_{image_path.name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(annotated).save(output_path)
    print(f"   Saved annotated image to: {output_path}")

    # Display result
    if not args.no_display:
        print("\n6. Displaying result...")
        plt.figure(figsize=(15, 10))
        plt.imshow(annotated)
        plt.axis('off')
        plt.title(f"Parking Lot Detection - {len(detections)} vehicles detected", fontsize=16)
        plt.tight_layout()
        plt.show()
        print("   Close the window to exit.")
    else:
        print("\n6. Display skipped (--no-display flag)")

    print("\n" + "=" * 60)
    print("Detection complete!")
    print("=" * 60)

    # Next steps
    print("\nNext steps:")
    print("  1. Try different confidence thresholds: -t 0.3 or -t 0.7")
    print("  2. Try different model sizes: -m small (faster) or -m base (more accurate)")
    print("  3. For better parking lot detection, fine-tune on your own dataset")
    print("     See PARKING_LOT_DETECTION.md for details on fine-tuning")
    print("  4. Use parking_lot_detector.py for batch processing:")
    print("     python parking_lot_detector.py images/ -o output/")


if __name__ == "__main__":
    main()
