# Documentation Troubleshooting Guide

## Common Issues and Solutions

### ❌ Error: "mkdocstrings: ecgen.data.datamodule could not be found"

**Problem:** The ECGEN package is not installed, so mkdocstrings cannot import the modules.

**Solution:**
```bash
# Install the ECGEN package first
pip install -e .

# Then try again
mkdocs serve
```

**Why:** The API documentation uses `mkdocstrings` which imports your Python code to extract docstrings. The package must be installed for this to work.

---

### ⚠️ Warning: "MkDocs 2.0 is incompatible with Material for MkDocs"

**Problem:** Version compatibility warning from Material theme.

**Solution:** This is just a warning and can be safely ignored. Your documentation will work fine.

**Alternative:** If you want to suppress it, you can use MkDocs 1.6.x:
```bash
pip install "mkdocs<2.0"
```

---

### ⚠️ Warning: "pages exist but are not included in nav"

**Problem:** Some documentation files aren't listed in the navigation.

**Solution:** This is expected for:
- `pulse2pulse_training.md` and `wandb_setup.md` (old files, kept for reference)
- Auto-generated API reference files

You can safely ignore these warnings.

---

### ❌ Error: "Address already in use"

**Problem:** Port 8000 is already being used by another process.

**Solution:**
```bash
# Option 1: Use a different port
mkdocs serve -a 127.0.0.1:8001

# Option 2: Kill the process using port 8000
lsof -ti:8000 | xargs kill -9
```

---

### ❌ Error: "No module named 'ecgen'"

**Problem:** ECGEN package is not installed.

**Solution:**
```bash
# Make sure you're in the ECGEN directory
cd /work/vajira/DL2026/ECGEN

# Install the package
pip install -e .

# Verify
python -c "import ecgen; print('✓ Works')"
```

---

### ❌ Build Error: "Could not collect 'ecgen.xxx'"

**Problem:** Module exists but can't be imported (usually due to syntax errors or missing dependencies).

**Solution:**
```bash
# Test if the module can be imported
python -c "from ecgen.models import pulse2pulse"

# If it fails, check for:
# 1. Syntax errors in the Python file
# 2. Missing dependencies
# 3. Import errors in the module

# Install any missing dependencies
pip install -e .
```

---

### 🔗 Warning: "contains an unrecognized relative link"

**Problem:** Some links in the documentation use directory paths instead of file paths.

**Solution:** These are informational warnings and don't break the documentation. The links will still work in most cases.

To fix them, change:
- `[Link](reference/)` → `[Link](reference/index.md)`

---

### 📁 API Reference Pages Are Empty

**Problem:** API reference shows no content.

**Causes:**
1. ECGEN package not installed
2. Modules have no docstrings
3. Import errors in the modules

**Solutions:**

**Check 1: Package installed?**
```bash
pip list | grep ecggen
# Should show: ecggen 0.1.0 /path/to/ECGEN/src
```

**Check 2: Can Python import it?**
```bash
python -c "from ecgen.models.pulse2pulse import Pulse2PulseGAN; print('✓')"
```

**Check 3: Do modules have docstrings?**
```bash
grep -r "\"\"\"" src/ecgen/models/pulse2pulse.py
```

---

### 🌐 GitHub Pages Shows 404

**Problem:** Documentation deployed but shows 404 error.

**Solutions:**

**Check 1: Is gh-pages branch created?**
- Go to repository → Branches
- Look for `gh-pages` branch

**Check 2: Are Pages enabled?**
- Go to Settings → Pages
- Source should be: Deploy from branch
- Branch should be: `gh-pages`

**Check 3: Wait and clear cache**
- Wait 5-10 minutes after deployment
- Clear browser cache
- Try incognito/private mode

**Check 4: Check the URL**
- Should be: `https://username.github.io/ECGEN`
- NOT: `https://username.github.io/ECGEN/docs`

---

### 🔄 GitHub Actions Workflow Fails

**Problem:** Documentation deployment fails in GitHub Actions.

**Solutions:**

**Check 1: View the error**
- Go to Actions tab
- Click on the failed workflow
- Read the error message

**Check 2: Common causes**
- Missing dependencies in `pyproject.toml`
- Syntax errors in `mkdocs.yml`
- Import errors in Python code
- Missing required files

**Check 3: Test locally first**
```bash
# This should work without errors
mkdocs build

# If it works locally but fails in CI, check:
# - All files are committed
# - .github/workflows/docs.yml is correct
```

---

### 🔍 Search Not Working

**Problem:** Search returns no results.

**Solution:**
```bash
# Rebuild the search index
mkdocs build --clean

# Check search plugin is enabled in mkdocs.yml
grep -A 2 "plugins:" mkdocs.yml
```

---

### 📱 Documentation Looks Broken on Mobile

**Problem:** Layout issues on mobile devices.

**Solution:** This shouldn't happen with Material theme. If it does:

1. Clear browser cache
2. Try a different mobile browser
3. Check if custom CSS is interfering
4. Verify Material theme is properly installed:
   ```bash
   pip install --upgrade mkdocs-material
   ```

---

## Quick Diagnostic Commands

Run these to check your setup:

```bash
# 1. Check ECGEN package
python -c "import ecgen; print('✓ ECGEN OK')"

# 2. Check MkDocs
mkdocs --version

# 3. Check Material theme
python -c "import material; print('✓ Material OK')"

# 4. Check mkdocstrings
python -c "import mkdocstrings; print('✓ mkdocstrings OK')"

# 5. Try building
mkdocs build --verbose

# 6. Check for Python errors
python -m py_compile src/ecgen/**/*.py
```

---

## Getting Help

If you're still stuck:

1. **Check the guides:**
   - [INSTALLATION_NOTES.md](INSTALLATION_NOTES.md)
   - [DOCUMENTATION_SETUP.md](DOCUMENTATION_SETUP.md)
   - [DOCS_MAINTENANCE.md](DOCS_MAINTENANCE.md)

2. **Check official docs:**
   - [MkDocs](https://www.mkdocs.org/)
   - [Material Theme](https://squidfunk.github.io/mkdocs-material/)
   - [mkdocstrings](https://mkdocstrings.github.io/)

3. **Common fixes:**
   ```bash
   # Nuclear option: reinstall everything
   pip uninstall mkdocs mkdocs-material mkdocstrings ecggen
   pip install -e .
   pip install -e ".[docs]"
   mkdocs serve
   ```

---

## Prevention Tips

To avoid issues:

1. ✅ Always install ECGEN package first: `pip install -e .`
2. ✅ Then install docs dependencies: `pip install -e ".[docs]"`
3. ✅ Test locally before pushing: `mkdocs build`
4. ✅ Keep dependencies updated: `pip install --upgrade mkdocs mkdocs-material`
5. ✅ Use the convenience script: `./docs_serve.sh`
