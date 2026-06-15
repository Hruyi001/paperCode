"""Quick script to verify checkpoint contains DSA module weights"""
import torch
import sys

ckpt_path = "checkpoint/pretrained_models/U1652/weights_end.pth"

try:
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    print(f"\nTotal keys in checkpoint: {len(ckpt.keys())}")
    
    # Check for DSA module (proj layer)
    proj_keys = [k for k in ckpt.keys() if 'proj' in k.lower()]
    print(f"\nDSA module (proj) keys found: {len(proj_keys)}")
    if proj_keys:
        print("  DSA module keys:")
        for k in proj_keys[:10]:  # Show first 10
            print(f"    - {k}")
        if len(proj_keys) > 10:
            print(f"    ... and {len(proj_keys) - 10} more")
    else:
        print("  ⚠️  WARNING: No DSA module (proj) keys found!")
    
    # Check for model structure
    model_keys = [k for k in ckpt.keys() if 'model_1' in k]
    print(f"\nModel structure keys (model_1): {len(model_keys)}")
    
    # Check for convnext backbone
    convnext_keys = [k for k in ckpt.keys() if 'convnext' in k.lower()]
    print(f"ConvNext backbone keys: {len(convnext_keys)}")
    
    print("\n✓ Checkpoint file is valid and contains model weights")
    if proj_keys:
        print("✓ DSA module weights are present")
    else:
        print("⚠️  DSA module weights may be missing")
        
except Exception as e:
    print(f"✗ Error loading checkpoint: {e}")
    sys.exit(1)
