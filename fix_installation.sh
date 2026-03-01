#!/bin/bash
# Fix installation issues for ECGEN

echo "Fixing PyYAML installation issue..."

# Option 1: Try force reinstall
echo "Attempting to reinstall PyYAML..."
pip install --ignore-installed PyYAML

# Install the package
echo "Installing ECGEN package..."
pip install -e .

# Verify installation
echo ""
echo "Verifying installation..."
python verify_reorganization.py

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. If verification passed, you're ready to use the new structure"
echo "2. Update your training scripts to use new imports (see MIGRATION_GUIDE.md)"
echo "3. Test training: python scripts/train/train_vae_mimic.py --help"
