#!/usr/bin/env python3
"""
Test script to verify that parking lot detection is working correctly.

This script checks:
1. Required dependencies are installed
2. Can import the detector classes
3. Can initialize models
4. System is ready for parking lot detection

Run: python test_installation.py
"""

import sys

def check_python_version():
    """Check Python version is 3.9+"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.9+)")
        return False

def check_dependencies():
    """Check required packages are installed"""
    print("\nChecking dependencies...")

    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'supervision': 'Supervision',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
    }

    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            all_ok = False

    return all_ok

def check_rfdetr():
    """Check RF-DETR package"""
    print("\nChecking RF-DETR...")
    try:
        import rfdetr
        print(f"  ✓ RF-DETR package found")

        # Try to import model classes
        from rfdetr import RFDETRSegPreview, RFDETRMedium
        print(f"  ✓ Model classes available")

        return True
    except ImportError as e:
        print(f"  ✗ RF-DETR not found: {e}")
        print(f"     Run: pip install -e .")
        return False

def check_detector_scripts():
    """Check our detector scripts exist"""
    print("\nChecking detector scripts...")
    from pathlib import Path

    scripts = [
        'parking_lot_segmentation_detector.py',
        'example_parking_segmentation.py',
        'parking_lot_detector.py',
        'example_parking_detection.py',
    ]

    all_ok = True
    for script in scripts:
        if Path(script).exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} - NOT FOUND")
            all_ok = False

    return all_ok

def check_cuda():
    """Check CUDA/GPU availability"""
    print("\nChecking GPU/CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
            return "gpu"
        else:
            print(f"  ⚠ CUDA not available (will use CPU)")
            print(f"    CPU inference will be slower but still works")
            return "cpu"
    except:
        print(f"  ⚠ Could not check CUDA")
        return "unknown"

def test_model_init():
    """Try to initialize the segmentation model"""
    print("\nTesting model initialization...")
    try:
        from parking_lot_segmentation_detector import ParkingLotSegmentationDetector

        print("  Initializing model (this may download weights)...")
        detector = ParkingLotSegmentationDetector(
            use_segmentation=True,
            optimize=False  # Skip optimization for faster test
        )
        print(f"  ✓ Model initialized successfully")
        print(f"  ✓ Model device: {detector.device}")
        print(f"  ✓ Segmentation enabled: {detector.use_segmentation}")

        return True
    except Exception as e:
        print(f"  ✗ Model initialization failed: {e}")
        return False

def main():
    print("=" * 60)
    print("PARKING LOT DETECTION - INSTALLATION TEST")
    print("=" * 60)

    results = []

    # Run all checks
    results.append(("Python Version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("RF-DETR Package", check_rfdetr()))
    results.append(("Detector Scripts", check_detector_scripts()))
    device = check_cuda()
    results.append(("Model Initialization", test_model_init()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = all(result for _, result in results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:.<40} {status}")

    print("=" * 60)

    if all_passed:
        print("\n🎉 All checks passed! You're ready to use parking lot detection.")
        print("\nNext steps:")
        print("  1. Get a satellite image of a parking lot")
        print("  2. Run: python example_parking_segmentation.py your_image.jpg")
        print("\nSee GETTING_STARTED.md for detailed instructions.")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -e .")
        print("  - Install PyTorch: pip install torch torchvision")
        print("  - Install Supervision: pip install supervision")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
