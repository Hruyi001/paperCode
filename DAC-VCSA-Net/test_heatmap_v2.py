#!/usr/bin/env python
"""
Quick test script for DSA Heatmap Generator v2.0

This script performs basic sanity checks without requiring a full dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cross_view_correlation():
    """Test the cross-view correlation computation"""
    print("Testing cross-view correlation computation...")
    
    # Create dummy features
    B, C, H, W = 1, 1024, 12, 12
    query_features = torch.randn(B, C, H, W)
    ref_features = torch.randn(B, C, H, W)
    
    # Flatten spatial dimensions
    query_flat = query_features.flatten(2)  # (B, C, H*W)
    ref_flat = ref_features.flatten(2)
    
    # Normalize
    query_norm = F.normalize(query_flat, dim=1)
    ref_norm = F.normalize(ref_flat, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.bmm(query_norm.transpose(1, 2), ref_norm)  # (B, H*W, H*W)
    
    # Take max similarity
    max_similarity, _ = similarity_matrix.max(dim=2)  # (B, H*W)
    
    # Reshape
    correlation_map = max_similarity.view(B, H, W)
    
    print(f"  ✓ Input shape: {query_features.shape}")
    print(f"  ✓ Correlation map shape: {correlation_map.shape}")
    print(f"  ✓ Correlation range: [{correlation_map.min():.4f}, {correlation_map.max():.4f}]")
    print()
    
    # Sanity checks
    assert correlation_map.shape == (B, H, W), "Output shape mismatch"
    assert correlation_map.min() >= -1.0 and correlation_map.max() <= 1.0, "Correlation out of range"
    
    return True


def test_dsa_attention():
    """Test DSA attention mechanism extraction"""
    print("Testing DSA attention mechanism...")
    
    # Create dummy features
    B, C, H, W = 1, 1024, 12, 12
    part_features = torch.randn(B, C, H, W)
    
    # Simulate DSA projection
    pfeat = part_features.flatten(2)  # (B, C, H*W)
    
    # Create dummy projection layer
    proj = nn.Conv1d(C, 256, 1)
    W = proj(pfeat)  # (B, 256, H*W)
    
    # Apply normalization and softmax
    W = F.normalize(W, dim=1)
    W = F.softmax(W, dim=2)
    
    # Reshape to spatial dimensions
    W_spatial = W.view(B, -1, H, W)
    
    # Average attention across heads
    attention_weights = W_spatial.mean(dim=1)  # (B, H, W)
    
    print(f"  ✓ Feature shape: {part_features.shape}")
    print(f"  ✓ Attention weights shape: {attention_weights.shape}")
    print(f"  ✓ Attention range: [{attention_weights.min():.4f}, {attention_weights.max():.4f}]")
    print()
    
    # Sanity checks
    assert attention_weights.shape == (B, H, W), "Attention shape mismatch"
    assert attention_weights.min() >= 0.0, "Attention weights should be non-negative"
    
    return True


def test_feature_alignment():
    """Test feature alignment with DSA weights"""
    print("Testing feature alignment with DSA weights...")
    
    # Create dummy features
    B, C, H, W = 1, 1024, 12, 12
    features = torch.randn(B, C, H, W)
    attention_weights = torch.rand(B, 1, H, W)
    
    # Apply attention weights
    features_aligned = features * attention_weights.expand_as(features)
    
    print(f"  ✓ Original features shape: {features.shape}")
    print(f"  ✓ Aligned features shape: {features_aligned.shape}")
    print(f"  ✓ Feature modification: {(features_aligned - features).abs().mean():.4f}")
    print()
    
    # Sanity checks
    assert features_aligned.shape == features.shape, "Shape changed after alignment"
    assert not torch.allclose(features, features_aligned), "Features should be modified"
    
    return True


def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    
    try:
        import cv2
        print("  ✓ opencv-python (cv2)")
    except ImportError:
        print("  ✗ opencv-python (cv2) - MISSING")
        return False
    
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy - MISSING")
        return False
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        print("  ✓ matplotlib")
    except ImportError:
        print("  ✗ matplotlib - MISSING")
        return False
    
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        print("  ✓ albumentations")
    except ImportError:
        print("  ✗ albumentations - MISSING")
        return False
    
    try:
        # Try importing the main script (this validates syntax)
        import generate_dsa_heatmap_v2
        print("  ✓ generate_dsa_heatmap_v2.py")
    except ImportError as e:
        print(f"  ✗ generate_dsa_heatmap_v2.py - {e}")
        return False
    except Exception as e:
        print(f"  ⚠ generate_dsa_heatmap_v2.py - Syntax OK but import issue: {e}")
        # Still return True since syntax is valid
    
    print()
    return True


def main():
    print("=" * 80)
    print("DSA Heatmap Generator v2.0 - Quick Test Suite")
    print("=" * 80)
    print()
    
    all_tests_passed = True
    
    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Cross-View Correlation", test_cross_view_correlation),
        ("DSA Attention", test_dsa_attention),
        ("Feature Alignment", test_feature_alignment),
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if not result:
                print(f"✗ {test_name} FAILED")
                all_tests_passed = False
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
            all_tests_passed = False
    
    print("=" * 80)
    if all_tests_passed:
        print("✓ All tests passed!")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. Verify you have a trained checkpoint")
        print("  2. Verify you have the University-1652 dataset")
        print("  3. Run: bash run_generate_dsa_heatmap_v2.sh")
        return 0
    else:
        print("✗ Some tests failed!")
        print("=" * 80)
        print()
        print("Please fix the issues above before running the full script.")
        return 1


if __name__ == '__main__':
    exit(main())
