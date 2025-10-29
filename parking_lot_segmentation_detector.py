#!/usr/bin/env python3
"""
Parking Lot Segmentation Detection for Satellite Images using RF-DETR-Seg

This script uses the RF-DETR segmentation model to detect and segment parking lots
in satellite imagery. Segmentation provides pixel-level masks which are much better
for identifying parking lot regions compared to just bounding boxes.

The segmentation model can:
- Identify parking lot boundaries precisely
- Segment individual parking spaces
- Detect vehicles with pixel-perfect masks
- Calculate accurate areas and occupancy rates
"""

import os
import argparse
from typing import List, Optional, Union, Tuple, Dict
from pathlib import Path

import torch
import numpy as np
import supervision as sv
from PIL import Image
import cv2

from rfdetr import RFDETRSegPreview, RFDETRSmall, RFDETRMedium, RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES


class ParkingLotSegmentationDetector:
    """
    A segmentation-based detector for parking lots in satellite imagery using RF-DETR-Seg.
    Provides pixel-level masks for accurate parking lot and vehicle detection.
    """

    def __init__(
        self,
        use_segmentation: bool = True,
        model_size: str = "seg-preview",
        pretrain_weights: Optional[str] = None,
        device: Optional[str] = None,
        optimize: bool = True,
        threshold: float = 0.5,
        target_classes: Optional[List[str]] = None,
    ):
        """
        Initialize the parking lot segmentation detector.

        Args:
            use_segmentation: Whether to use segmentation model (True) or detection only (False)
            model_size: Model size ('seg-preview' for segmentation, 'small'/'medium'/'base' for detection)
            pretrain_weights: Path to custom weights (if fine-tuned on parking lots)
            device: Device to run on ('cuda', 'cpu', or None for auto)
            optimize: Whether to optimize the model for faster inference
            threshold: Confidence threshold for detections
            target_classes: List of class names to detect (None = all classes)
        """
        self.threshold = threshold
        self.target_classes = target_classes or ["car", "truck", "bus"]
        self.use_segmentation = use_segmentation

        # Load model based on size
        if use_segmentation or model_size == "seg-preview":
            print("Loading RF-DETR-Seg-Preview model (with segmentation)...")
            self.model = RFDETRSegPreview(pretrain_weights=pretrain_weights)
            self.use_segmentation = True
        else:
            print(f"Loading RF-DETR-{model_size.upper()} model (detection only)...")
            if model_size.lower() == "small":
                self.model = RFDETRSmall(pretrain_weights=pretrain_weights)
            elif model_size.lower() == "medium":
                self.model = RFDETRMedium(pretrain_weights=pretrain_weights)
            elif model_size.lower() == "base":
                self.model = RFDETRBase(pretrain_weights=pretrain_weights)
            else:
                raise ValueError(f"Unknown model size: {model_size}")

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.model.to(device)

        # Optimize for inference
        if optimize:
            print("Optimizing model for inference...")
            try:
                self.model.optimize_for_inference()
            except Exception as e:
                print(f"Warning: Could not optimize model: {e}")

        # Create class ID filter
        if target_classes:
            self.target_class_ids = [
                i for i, name in enumerate(COCO_CLASSES)
                if name.lower() in [tc.lower() for tc in target_classes]
            ]
            print(f"Filtering for classes: {target_classes} (IDs: {self.target_class_ids})")
        else:
            self.target_class_ids = None

        print(f"Model loaded on {device}")
        print(f"Segmentation enabled: {self.use_segmentation}")

    def detect(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        threshold: Optional[float] = None,
    ) -> sv.Detections:
        """
        Run detection/segmentation on a single image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            threshold: Confidence threshold (uses default if None)

        Returns:
            supervision.Detections object with detected objects (and masks if segmentation enabled)
        """
        threshold = threshold or self.threshold

        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Run detection
        detections = self.model.predict(image, threshold=threshold)

        # Filter by target classes if specified
        if self.target_class_ids is not None and len(detections) > 0:
            mask = np.isin(detections.class_id, self.target_class_ids)
            detections = detections[mask]

        return detections

    def detect_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        threshold: Optional[float] = None,
    ) -> List[sv.Detections]:
        """
        Run detection/segmentation on multiple images.

        Args:
            images: List of input images
            threshold: Confidence threshold (uses default if None)

        Returns:
            List of supervision.Detections objects
        """
        threshold = threshold or self.threshold

        # Load images if paths
        loaded_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                loaded_images.append(Image.open(img).convert("RGB"))
            else:
                loaded_images.append(img)

        # Run batch detection
        detections_list = self.model.predict(loaded_images, threshold=threshold)

        # Filter by target classes if specified
        if self.target_class_ids is not None:
            filtered_detections = []
            for detections in detections_list:
                if len(detections) > 0:
                    mask = np.isin(detections.class_id, self.target_class_ids)
                    filtered_detections.append(detections[mask])
                else:
                    filtered_detections.append(detections)
            return filtered_detections

        return detections_list

    def annotate_image(
        self,
        image: Union[Image.Image, np.ndarray],
        detections: sv.Detections,
        show_labels: bool = True,
        show_confidence: bool = True,
        show_masks: bool = True,
        mask_opacity: float = 0.5,
    ) -> np.ndarray:
        """
        Annotate an image with detection/segmentation results.

        Args:
            image: Input image
            detections: Detection results
            show_labels: Whether to show class labels
            show_confidence: Whether to show confidence scores
            show_masks: Whether to show segmentation masks (if available)
            mask_opacity: Opacity of segmentation masks (0-1)

        Returns:
            Annotated image as numpy array
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Create annotated image
        annotated = image.copy()

        # Draw masks if available and requested
        if show_masks and self.use_segmentation and len(detections) > 0 and detections.mask is not None:
            mask_annotator = sv.MaskAnnotator(opacity=mask_opacity)
            annotated = mask_annotator.annotate(annotated, detections)

        # Draw bounding boxes
        box_annotator = sv.BoxAnnotator(thickness=2)
        annotated = box_annotator.annotate(annotated, detections)

        # Draw labels if requested
        if show_labels and len(detections) > 0:
            labels = []
            for class_id, confidence in zip(detections.class_id, detections.confidence):
                class_name = COCO_CLASSES[class_id]
                if show_confidence:
                    labels.append(f"{class_name} {confidence:.2f}")
                else:
                    labels.append(class_name)

            label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
            annotated = label_annotator.annotate(annotated, detections, labels)

        return annotated

    def process_and_save(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        show_labels: bool = True,
        show_confidence: bool = True,
        show_masks: bool = True,
        mask_opacity: float = 0.5,
        threshold: Optional[float] = None,
    ) -> Tuple[sv.Detections, np.ndarray]:
        """
        Process an image and save the annotated result.

        Args:
            input_path: Path to input image
            output_path: Path to save annotated image
            show_labels: Whether to show class labels
            show_confidence: Whether to show confidence scores
            show_masks: Whether to show segmentation masks
            mask_opacity: Opacity of segmentation masks (0-1)
            threshold: Confidence threshold (uses default if None)

        Returns:
            Tuple of (detections, annotated_image)
        """
        # Load image
        image = Image.open(input_path).convert("RGB")

        # Run detection
        detections = self.detect(image, threshold=threshold)

        # Annotate
        annotated = self.annotate_image(
            image, detections, show_labels, show_confidence, show_masks, mask_opacity
        )

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(annotated).save(output_path)

        print(f"Processed {input_path}")
        print(f"  Found {len(detections)} objects")
        if self.use_segmentation and len(detections) > 0 and detections.mask is not None:
            print(f"  Segmentation masks: {detections.mask.shape}")
        print(f"  Saved to {output_path}")

        return detections, annotated

    def get_statistics(self, detections: sv.Detections) -> Dict:
        """
        Get statistics about detections including segmentation metrics.

        Args:
            detections: Detection results

        Returns:
            Dictionary with detection statistics
        """
        stats = {
            "total_detections": len(detections),
            "detections_by_class": {},
            "avg_confidence": float(np.mean(detections.confidence)) if len(detections) > 0 else 0.0,
            "max_confidence": float(np.max(detections.confidence)) if len(detections) > 0 else 0.0,
            "min_confidence": float(np.min(detections.confidence)) if len(detections) > 0 else 0.0,
        }

        # Count by class
        if len(detections) > 0:
            for class_id in np.unique(detections.class_id):
                class_name = COCO_CLASSES[class_id]
                count = np.sum(detections.class_id == class_id)
                stats["detections_by_class"][class_name] = int(count)

        # Add segmentation statistics if available
        if self.use_segmentation and len(detections) > 0 and detections.mask is not None:
            total_pixels = detections.mask[0].size
            mask_areas = []
            mask_coverage = []

            for i, mask in enumerate(detections.mask):
                area = np.sum(mask)
                mask_areas.append(int(area))
                mask_coverage.append(float(area / total_pixels * 100))

            stats["segmentation"] = {
                "total_mask_area": int(np.sum(mask_areas)),
                "avg_mask_area": float(np.mean(mask_areas)) if mask_areas else 0.0,
                "mask_areas": mask_areas,
                "coverage_percentages": mask_coverage,
                "total_coverage_percentage": float(np.sum(mask_coverage))
            }

        return stats

    def calculate_parking_area(
        self,
        detections: sv.Detections,
        image_size: Tuple[int, int],
        pixel_to_meter_ratio: Optional[float] = None
    ) -> Dict:
        """
        Calculate parking lot area statistics from segmentation masks.

        Args:
            detections: Detection results with masks
            image_size: (width, height) of the image in pixels
            pixel_to_meter_ratio: Optional conversion ratio (meters per pixel)

        Returns:
            Dictionary with area calculations
        """
        if not self.use_segmentation or detections.mask is None or len(detections) == 0:
            return {"error": "Segmentation masks not available"}

        width, height = image_size
        total_image_pixels = width * height

        # Calculate total parking lot area (union of all masks)
        combined_mask = np.zeros((height, width), dtype=bool)
        for mask in detections.mask:
            combined_mask = combined_mask | mask

        parking_area_pixels = np.sum(combined_mask)
        parking_coverage = parking_area_pixels / total_image_pixels * 100

        result = {
            "total_parking_area_pixels": int(parking_area_pixels),
            "parking_coverage_percentage": float(parking_coverage),
            "individual_object_areas_pixels": [int(np.sum(mask)) for mask in detections.mask],
        }

        # Add metric measurements if ratio provided
        if pixel_to_meter_ratio is not None:
            area_sq_meters = parking_area_pixels * (pixel_to_meter_ratio ** 2)
            result["total_parking_area_sq_meters"] = float(area_sq_meters)
            result["individual_object_areas_sq_meters"] = [
                float(np.sum(mask) * (pixel_to_meter_ratio ** 2))
                for mask in detections.mask
            ]

        return result

    def extract_masks_as_images(
        self,
        image: Union[Image.Image, np.ndarray],
        detections: sv.Detections,
        output_dir: Union[str, Path],
    ) -> List[Path]:
        """
        Extract individual segmentation masks as separate image files.

        Args:
            image: Original image
            detections: Detection results with masks
            output_dir: Directory to save mask images

        Returns:
            List of paths to saved mask images
        """
        if not self.use_segmentation or detections.mask is None or len(detections) == 0:
            print("No segmentation masks available")
            return []

        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, (mask, class_id, confidence) in enumerate(
            zip(detections.mask, detections.class_id, detections.confidence)
        ):
            class_name = COCO_CLASSES[class_id]

            # Create masked image (original image with mask applied)
            masked_image = image.copy()
            masked_image[~mask] = 0  # Set non-mask pixels to black

            # Save masked image
            filename = f"mask_{i:03d}_{class_name}_{confidence:.2f}.png"
            output_path = output_dir / filename
            Image.fromarray(masked_image).save(output_path)

            # Also save binary mask
            binary_filename = f"mask_{i:03d}_{class_name}_binary.png"
            binary_output_path = output_dir / binary_filename
            binary_mask = (mask * 255).astype(np.uint8)
            Image.fromarray(binary_mask).save(binary_output_path)

            saved_paths.extend([output_path, binary_output_path])

        print(f"Saved {len(saved_paths)} mask images to {output_dir}")
        return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Detect and segment parking lots in satellite images using RF-DETR-Seg"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input image or directory of images"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output",
        help="Output directory for annotated images (default: output)"
    )
    parser.add_argument(
        "--no-segmentation",
        action="store_true",
        help="Use detection-only model (no segmentation masks)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="seg-preview",
        help="Model size (default: seg-preview for segmentation, or small/medium/base)"
    )
    parser.add_argument(
        "-w", "--weights",
        type=str,
        default=None,
        help="Path to custom weights (for fine-tuned models)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "-c", "--classes",
        type=str,
        nargs="+",
        default=["car", "truck", "bus"],
        help="Classes to detect (default: car truck bus)"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable model optimization"
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Don't show class labels"
    )
    parser.add_argument(
        "--no-confidence",
        action="store_true",
        help="Don't show confidence scores"
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Don't show segmentation masks in output"
    )
    parser.add_argument(
        "--mask-opacity",
        type=float,
        default=0.5,
        help="Opacity of segmentation masks 0-1 (default: 0.5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print detection statistics"
    )
    parser.add_argument(
        "--extract-masks",
        action="store_true",
        help="Extract individual masks as separate images"
    )

    args = parser.parse_args()

    # Create detector
    detector = ParkingLotSegmentationDetector(
        use_segmentation=not args.no_segmentation,
        model_size=args.model,
        pretrain_weights=args.weights,
        device=args.device,
        optimize=not args.no_optimize,
        threshold=args.threshold,
        target_classes=args.classes,
    )

    # Process input
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single image or directory
    if input_path.is_file():
        # Single image
        output_path = output_dir / f"annotated_{input_path.name}"
        detections, annotated = detector.process_and_save(
            input_path,
            output_path,
            show_labels=not args.no_labels,
            show_confidence=not args.no_confidence,
            show_masks=not args.no_masks,
            mask_opacity=args.mask_opacity,
        )

        if args.stats:
            print("\nDetection Statistics:")
            stats = detector.get_statistics(detections)
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        if isinstance(v, list):
                            print(f"    {k}: {len(v)} items")
                        else:
                            print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")

        if args.extract_masks:
            image = Image.open(input_path)
            mask_dir = output_dir / "masks"
            detector.extract_masks_as_images(image, detections, mask_dir)

    elif input_path.is_dir():
        # Directory of images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print(f"No images found in {input_path}")
            return

        print(f"\nProcessing {len(image_files)} images...")

        all_stats = []
        for img_file in image_files:
            output_path = output_dir / f"annotated_{img_file.name}"
            detections, _ = detector.process_and_save(
                img_file,
                output_path,
                show_labels=not args.no_labels,
                show_confidence=not args.no_confidence,
                show_masks=not args.no_masks,
                mask_opacity=args.mask_opacity,
            )

            if args.stats:
                all_stats.append(detector.get_statistics(detections))

            if args.extract_masks:
                image = Image.open(img_file)
                mask_dir = output_dir / "masks" / img_file.stem
                detector.extract_masks_as_images(image, detections, mask_dir)

        if args.stats:
            print("\n=== Overall Statistics ===")
            total_detections = sum(s["total_detections"] for s in all_stats)
            avg_confidence = np.mean([s["avg_confidence"] for s in all_stats if s["avg_confidence"] > 0])
            print(f"Total images processed: {len(image_files)}")
            print(f"Total detections: {total_detections}")
            print(f"Average confidence: {avg_confidence:.3f}")
            print(f"Average detections per image: {total_detections / len(image_files):.1f}")

    else:
        print(f"Error: {input_path} is not a file or directory")
        return

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
