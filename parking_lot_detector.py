#!/usr/bin/env python3
"""
Parking Lot Detection for Satellite Images using RF-DETR

This script uses the RF-DETR model to detect parking lots and vehicles in satellite imagery.
For best results, fine-tune the model on a parking lot dataset. The pre-trained COCO model
can detect cars, which can be used as a proxy for parking lot detection.
"""

import os
import argparse
from typing import List, Optional, Union, Tuple
from pathlib import Path

import torch
import numpy as np
import supervision as sv
from PIL import Image

from rfdetr import RFDETRMedium, RFDETRSmall, RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES


class ParkingLotDetector:
    """
    A detector for parking lots in satellite imagery using RF-DETR.
    """

    def __init__(
        self,
        model_size: str = "medium",
        pretrain_weights: Optional[str] = None,
        device: Optional[str] = None,
        optimize: bool = True,
        threshold: float = 0.5,
        target_classes: Optional[List[str]] = None,
    ):
        """
        Initialize the parking lot detector.

        Args:
            model_size: Model size to use ('small', 'medium', 'base')
            pretrain_weights: Path to custom weights (if fine-tuned on parking lots)
            device: Device to run on ('cuda', 'cpu', or None for auto)
            optimize: Whether to optimize the model for faster inference
            threshold: Confidence threshold for detections
            target_classes: List of class names to detect (None = all classes)
        """
        self.threshold = threshold
        self.target_classes = target_classes or ["car", "truck", "bus"]

        # Load model based on size
        print(f"Loading RF-DETR-{model_size.upper()} model...")
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
            self.model.optimize_for_inference()

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

    def detect(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        threshold: Optional[float] = None,
    ) -> sv.Detections:
        """
        Run detection on a single image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            threshold: Confidence threshold (uses default if None)

        Returns:
            supervision.Detections object with detected objects
        """
        threshold = threshold or self.threshold

        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Run detection
        detections = self.model.predict(image, threshold=threshold)

        # Filter by target classes if specified
        if self.target_class_ids is not None:
            mask = np.isin(detections.class_id, self.target_class_ids)
            detections = detections[mask]

        return detections

    def detect_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        threshold: Optional[float] = None,
    ) -> List[sv.Detections]:
        """
        Run detection on multiple images.

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
                mask = np.isin(detections.class_id, self.target_class_ids)
                filtered_detections.append(detections[mask])
            return filtered_detections

        return detections_list

    def annotate_image(
        self,
        image: Union[Image.Image, np.ndarray],
        detections: sv.Detections,
        show_labels: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Annotate an image with detection results.

        Args:
            image: Input image
            detections: Detection results
            show_labels: Whether to show class labels
            show_confidence: Whether to show confidence scores

        Returns:
            Annotated image as numpy array
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Create annotated image
        annotated = image.copy()

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
        threshold: Optional[float] = None,
    ) -> Tuple[sv.Detections, np.ndarray]:
        """
        Process an image and save the annotated result.

        Args:
            input_path: Path to input image
            output_path: Path to save annotated image
            show_labels: Whether to show class labels
            show_confidence: Whether to show confidence scores
            threshold: Confidence threshold (uses default if None)

        Returns:
            Tuple of (detections, annotated_image)
        """
        # Load image
        image = Image.open(input_path).convert("RGB")

        # Run detection
        detections = self.detect(image, threshold=threshold)

        # Annotate
        annotated = self.annotate_image(image, detections, show_labels, show_confidence)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(annotated).save(output_path)

        print(f"Processed {input_path}")
        print(f"  Found {len(detections)} objects")
        print(f"  Saved to {output_path}")

        return detections, annotated

    def get_statistics(self, detections: sv.Detections) -> dict:
        """
        Get statistics about detections.

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

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Detect parking lots and vehicles in satellite images using RF-DETR"
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
        "-m", "--model",
        type=str,
        choices=["small", "medium", "base"],
        default="medium",
        help="Model size (default: medium)"
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

    args = parser.parse_args()

    # Create detector
    detector = ParkingLotDetector(
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
        detections, _ = detector.process_and_save(
            input_path,
            output_path,
            show_labels=not args.no_labels,
            show_confidence=not args.no_confidence,
        )

        if args.stats:
            print("\nDetection Statistics:")
            stats = detector.get_statistics(detections)
            for key, value in stats.items():
                print(f"  {key}: {value}")

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
            )

            if args.stats:
                all_stats.append(detector.get_statistics(detections))

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
