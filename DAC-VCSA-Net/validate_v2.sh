#!/bin/bash

# Quick validation script for DSA Heatmap Generator v2.0

echo "=================================="
echo "DSA Heatmap v2.0 - Quick Validation"
echo "=================================="
echo ""

# Check if main script exists
echo "[1/4] Checking file existence..."
if [ ! -f "generate_dsa_heatmap_v2.py" ]; then
    echo "  ✗ generate_dsa_heatmap_v2.py not found!"
    exit 1
fi
echo "  ✓ generate_dsa_heatmap_v2.py"

if [ ! -f "run_generate_dsa_heatmap_v2.sh" ]; then
    echo "  ✗ run_generate_dsa_heatmap_v2.sh not found!"
    exit 1
fi
echo "  ✓ run_generate_dsa_heatmap_v2.sh"

if [ ! -f "README_HEATMAP_V2.md" ]; then
    echo "  ✗ README_HEATMAP_V2.md not found!"
    exit 1
fi
echo "  ✓ README_HEATMAP_V2.md"

if [ ! -f "COMPARISON_V1_VS_V2.md" ]; then
    echo "  ✗ COMPARISON_V1_VS_V2.md not found!"
    exit 1
fi
echo "  ✓ COMPARISON_V1_VS_V2.md"

echo ""

# Check Python syntax
echo "[2/4] Checking Python syntax..."
python3 -m py_compile generate_dsa_heatmap_v2.py 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ Python syntax is valid"
else
    echo "  ✗ Python syntax error detected!"
    exit 1
fi

echo ""

# Check key components in the script
echo "[3/4] Checking key components..."

if grep -q "class CrossViewHeatmapGenerator" generate_dsa_heatmap_v2.py; then
    echo "  ✓ CrossViewHeatmapGenerator class found"
else
    echo "  ✗ CrossViewHeatmapGenerator class not found!"
    exit 1
fi

if grep -q "compute_cross_view_correlation" generate_dsa_heatmap_v2.py; then
    echo "  ✓ compute_cross_view_correlation method found"
else
    echo "  ✗ compute_cross_view_correlation method not found!"
    exit 1
fi

if grep -q "def generate_heatmap" generate_dsa_heatmap_v2.py; then
    echo "  ✓ generate_heatmap function found"
else
    echo "  ✗ generate_heatmap function not found!"
    exit 1
fi

if grep -q "def load_image_pair" generate_dsa_heatmap_v2.py; then
    echo "  ✓ load_image_pair function found"
else
    echo "  ✗ load_image_pair function not found!"
    exit 1
fi

echo ""

# Check documentation
echo "[4/4] Checking documentation..."

README_LINES=$(wc -l < README_HEATMAP_V2.md)
if [ "$README_LINES" -gt 100 ]; then
    echo "  ✓ README is comprehensive ($README_LINES lines)"
else
    echo "  ⚠ README might be incomplete ($README_LINES lines)"
fi

COMPARISON_LINES=$(wc -l < COMPARISON_V1_VS_V2.md)
if [ "$COMPARISON_LINES" -gt 50 ]; then
    echo "  ✓ Comparison document is detailed ($COMPARISON_LINES lines)"
else
    echo "  ⚠ Comparison document might be incomplete ($COMPARISON_LINES lines)"
fi

echo ""
echo "=================================="
echo "✓ Validation completed successfully!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Update checkpoint path in run_generate_dsa_heatmap_v2.sh"
echo "  2. Update dataset path in run_generate_dsa_heatmap_v2.sh"
echo "  3. Run: bash run_generate_dsa_heatmap_v2.sh"
echo ""
echo "Documentation:"
echo "  - README_HEATMAP_V2.md: Full user guide"
echo "  - COMPARISON_V1_VS_V2.md: Differences from v1.0"
echo ""
