# Installation Notes for Documentation

## Important: Install Package First

Before serving the documentation, you **must** install the ECGEN package in editable mode so that `mkdocstrings` can import the modules:

```bash
# Install ECGEN package
pip install -e .

# Install documentation dependencies
pip install -e ".[docs]"

# Now you can serve the docs
mkdocs serve
```

## Why This Is Needed

The API reference documentation uses `mkdocstrings` which imports your Python modules to extract docstrings. If the package isn't installed, the imports will fail and you'll get errors like:

```
ERROR - mkdocstrings: ecgen.data.datamodule could not be found
```

## Quick Fix

If you see import errors when running `mkdocs serve`:

```bash
# Make sure you're in the ECGEN directory
cd /work/vajira/DL2026/ECGEN

# Install the package
pip install -e .

# Try again
mkdocs serve
```

## Full Installation Steps

1. **Install ECGEN package:**
   ```bash
   pip install -e .
   ```

2. **Install documentation tools:**
   ```bash
   pip install -e ".[docs]"
   ```

3. **Verify installation:**
   ```bash
   python -c "import ecgen; print('✓ ECGEN installed')"
   mkdocs --version
   ```

4. **Serve documentation:**
   ```bash
   mkdocs serve
   ```

## Troubleshooting

### Still Getting Import Errors?

1. Check that the package is installed:
   ```bash
   pip list | grep ecggen
   ```

2. Try reinstalling:
   ```bash
   pip uninstall ecggen
   pip install -e .
   ```

3. Verify Python can import it:
   ```bash
   python -c "from ecgen.models import pulse2pulse; print('✓ Works')"
   ```

### MkDocs Version Warning

You may see a warning about MkDocs 2.0. This is safe to ignore - the documentation will still work correctly with MkDocs 1.x and Material theme.

## Alternative: Documentation Without API Reference

If you just want to view the documentation without the API reference, you can:

1. Comment out the API reference pages in `mkdocs.yml`
2. Or simply ignore the warnings and view the user guides

The user guides, tutorials, and other documentation will work fine without the package being installed.
