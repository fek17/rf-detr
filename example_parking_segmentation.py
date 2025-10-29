#!/usr/bin/env python3
"""
Example script demonstrating parking lot segmentation for satellite images.

This shows how to use the RF-DETR-Seg model to get pixel-level masks of parking
lots and vehicles, which is much more accurate than just bounding boxes.

Usage:
    python example_parking_segmentation.py path/to/satellite_image.jpg
"""

import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import our segmentation detector
from parking_lot_segmentation_detector import ParkingLotSegmentationDetector
from rfdetr.util.coco_classes import COCO_CLASSES


def visualize_comparison(image, detections, detector):
    """
    Create a comparison visualization showing:
    1. Original image
    2. Bounding boxes only
    3. Segmentation masks only
    4. Combined (boxes + masks)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # 1. Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Satellite Image", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # 2. Bounding boxes only
    img_boxes = detector.annotate_image(
        image, detections, show_labels=True, show_confidence=True, show_masks=False
    )
    axes[0, 1].imshow(img_boxes)
    axes[0, 1].set_title(f"Bounding Boxes Only\n{len(detections)} detections", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # 3. Segmentation masks only
    if detector.use_segmentation and detections.mask is not None and len(detections) > 0:
        img_masks = detector.annotate_image(
            image, detections, show_labels=False, show_confidence=False, show_masks=True, mask_opacity=0.7
        )
        axes[1, 0].imshow(img_masks)
        axes[1, 0].set_title("Segmentation Masks Only\nPixel-level precision", fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        # 4. Combined
        img_combined = detector.annotate_image(
            image, detections, show_labels=True, show_confidence=True, show_masks=True, mask_opacity=0.5
        )
        axes[1, 1].imshow(img_combined)
        axes[1, 1].set_title("Combined: Boxes + Masks\nComplete detection", fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
    else:
        axes[1, 0].text(0.5, 0.5, "Segmentation not available", ha='center', va='center', fontsize=14)
        axes[1, 0].axis('off')
        axes[1, 1].text(0.5, 0.5, "Segmentation not available", ha='center', va='center', fontsize=14)
        axes[1, 1].axis('off')

    plt.tight_layout()
    return fig


def print_detailed_stats(detections, detector, image_size):
    """Print detailed statistics about the detections."""
    stats = detector.get_statistics(detections)

    print("\n" + "=" * 60)
    print("DETECTION STATISTICS")
    print("=" * 60)

    print(f"\nTotal Detections: {stats['total_detections']}")

    if stats['total_detections'] > 0:
        print(f"\nConfidence Scores:")
        print(f"  Average: {stats['avg_confidence']:.3f}")
        print(f"  Range: {stats['min_confidence']:.3f} - {stats['max_confidence']:.3f}")

        if stats['detections_by_class']:
            print(f"\nDetections by Class:")
            for class_name, count in stats['detections_by_class'].items():
                print(f"  {class_name}: {count}")

        # Print individual detections
        print(f"\nIndividual Detections:")
        for i, (class_id, conf, bbox) in enumerate(
            zip(detections.class_id, detections.confidence, detections.xyxy)
        ):
            class_name = COCO_CLASSES[class_id]
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            print(f"  {i+1}. {class_name:15s} conf={conf:.3f} bbox=[{x1:4.0f},{y1:4.0f},{x2:4.0f},{y2:4.0f}] size={w:4.0f}x{h:4.0f}")

        # Segmentation statistics
        if 'segmentation' in stats:
            seg = stats['segmentation']
            print(f"\n" + "=" * 60)
            print("SEGMENTATION STATISTICS")
            print("=" * 60)

            print(f"\nTotal Segmented Area: {seg['total_mask_area']:,} pixels")
            print(f"Average Mask Area: {seg['avg_mask_area']:.0f} pixels")
            print(f"Total Coverage: {seg['total_coverage_percentage']:.2f}% of image")

            print(f"\nIndividual Mask Areas:")
            for i, (area, coverage, class_id) in enumerate(
                zip(seg['mask_areas'], seg['coverage_percentages'], detections.class_id)
            ):
                class_name = COCO_CLASSES[class_id]
                print(f"  {i+1}. {class_name:15s} area={area:7,} pixels ({coverage:.3f}% of image)")

            # Calculate parking area
            area_stats = detector.calculate_parking_area(detections, image_size)
            print(f"\n" + "=" * 60)
            print("PARKING LOT AREA ANALYSIS")
            print("=" * 60)
            print(f"\nTotal Parking Area: {area_stats['total_parking_area_pixels']:,} pixels")
            print(f"Parking Coverage: {area_stats['parking_coverage_percentage']:.2f}% of image")

            # Estimate number of parking spaces (rough estimate)
            # Assuming average car is ~15 sq meters, parking space is ~20 sq meters
            if len(detections) > 0:
                avg_vehicle_area = np.mean(seg['mask_areas'])
                # Very rough estimate: parking lot is usually 2-3x the area of cars
                estimated_spaces = int(area_stats['total_parking_area_pixels'] / avg_vehicle_area * 0.7)
                print(f"\nRough Estimates:")
                print(f"  Average vehicle area: {avg_vehicle_area:.0f} pixels")
                print(f"  Estimated total parking spaces: ~{estimated_spaces}")
                print(f"  Occupied spaces: {len(detections)}")
                if estimated_spaces > 0:
                    occupancy = len(detections) / estimated_spaces * 100
                    print(f"  Estimated occupancy rate: ~{occupancy:.0f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Example: Parking lot segmentation in satellite images"
    )
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to satellite image"
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
        help="Don't display the visualization"
    )
    parser.add_argument(
        "-c", "--classes",
        type=str,
        nargs="+",
        default=None,
        help="Classes to detect (default: all classes)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RF-DETR PARKING LOT SEGMENTATION EXAMPLE")
    print("=" * 60)

    # Check for image
    if args.image_path is None:
        print("\n❌ No image provided!")
        print("\nUsage:")
        print("  python example_parking_segmentation.py path/to/satellite_image.jpg")
        print("\nThis script demonstrates pixel-level segmentation for parking lot detection.")
        print("Segmentation provides much more accurate boundaries than bounding boxes alone.")
        print("\nYou can get satellite images from:")
        print("  - Google Earth Pro (free desktop app)")
        print("  - Bing Maps or OpenStreetMap")
        print("  - Public datasets (DOTA, DIOR, xView, etc.)")
        return

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"\n❌ Error: Image not found at {image_path}")
        return

    # Initialize detector with segmentation
    print(f"\n1. Initializing RF-DETR-Seg model...")
    print("   This model provides pixel-level segmentation masks.")
    detector = ParkingLotSegmentationDetector(
        use_segmentation=True,
        model_size="seg-preview",
        threshold=args.threshold,
        target_classes=args.classes,
        optimize=True
    )

    # Load image
    print(f"\n2. Loading image from {image_path}...")
    image = Image.open(image_path).convert("RGB")
    image_size = image.size
    print(f"   Image size: {image_size[0]}x{image_size[1]} pixels")

    # Run segmentation
    print(f"\n3. Running segmentation (threshold={args.threshold})...")
    detections = detector.detect(image, threshold=args.threshold)

    if len(detections) == 0:
        print("\n   ⚠️  No detections found!")
        print("   Try:")
        print("     - Lowering the threshold: -t 0.3")
        print("     - Checking if image shows parking lots/vehicles")
        print("     - Fine-tuning the model on your specific satellite imagery")
        return

    print(f"   ✓ Found {len(detections)} objects!")

    # Print detailed statistics
    print_detailed_stats(detections, detector, image_size)

    # Save results
    print(f"\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save annotated image
    annotated = detector.annotate_image(
        image, detections, show_labels=True, show_confidence=True,
        show_masks=True, mask_opacity=0.5
    )
    output_path = output_dir / f"segmented_{image_path.name}"
    Image.fromarray(annotated).save(output_path)
    print(f"\n✓ Saved annotated image to: {output_path}")

    # Extract individual masks
    if detector.use_segmentation and detections.mask is not None:
        mask_dir = output_dir / "masks"
        mask_paths = detector.extract_masks_as_images(image, detections, mask_dir)
        print(f"✓ Saved {len(mask_paths)} individual mask images to: {mask_dir}")

    # Create comparison visualization
    if not args.no_display:
        print(f"\n4. Creating comparison visualization...")
        fig = visualize_comparison(np.array(image), detections, detector)

        # Save comparison
        comparison_path = output_dir / f"comparison_{image_path.name}"
        fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved comparison to: {comparison_path}")

        print(f"\n5. Displaying results...")
        print("   Close the window to exit.")
        plt.show()
    else:
        print("\n   Display skipped (--no-display flag)")

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)

    print("\n💡 Next Steps:")
    print("   1. Try different thresholds: -t 0.3 (more detections) or -t 0.7 (fewer false positives)")
    print("   2. For custom parking lot detection, fine-tune on your own dataset")
    print("   3. See PARKING_LOT_DETECTION.md for fine-tuning guide")
    print("   4. Use parking_lot_segmentation_detector.py for batch processing")


if __name__ == "__main__":
    main()
