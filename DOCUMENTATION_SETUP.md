# ECGEN Documentation Setup Guide

Complete guide for setting up and using the ECGEN documentation system.

## 🎯 What You Get

Your ECGEN repository now has:

✅ **Professional Documentation Site** with Material Design theme  
✅ **Automatic API Reference** generated from code docstrings  
✅ **GitHub Pages Deployment** via GitHub Actions  
✅ **Local Development Server** with live reload  
✅ **Search Functionality** built-in  
✅ **Mobile-Friendly** responsive design  
✅ **Dark/Light Mode** theme toggle  

## 📦 Installation

### 1. Install ECGEN Package (Required!)

**Important:** You must install the ECGEN package first so the API documentation can import the modules:

```bash
cd /work/vajira/DL2026/ECGEN
pip install -e .
```

### 2. Install Documentation Dependencies

```bash
pip install -e ".[docs]"
```

This installs:
- `mkdocs` - Static site generator
- `mkdocs-material` - Material Design theme
- `mkdocstrings[python]` - API documentation from docstrings

### 3. Verify Installation

```bash
# Check MkDocs
mkdocs --version

# Check ECGEN package
python -c "import ecgen; print('✓ ECGEN installed')"
```

Expected output: `mkdocs, version 1.5.x` and `✓ ECGEN installed`

## 🚀 Usage

### Serve Documentation Locally

**Option 1: Using MkDocs directly**
```bash
mkdocs serve
```

**Option 2: Using the convenience script**
```bash
./docs_serve.sh
```

Then open http://127.0.0.1:8000 in your browser.

The server will automatically reload when you save changes to documentation files.

### Build Static Site

```bash
mkdocs build
```

Output will be in the `site/` directory. You can open `site/index.html` in a browser.

### Deploy to GitHub Pages

**Option 1: Automatic (Recommended)**

Just push to your main/master branch:
```bash
git add .
git commit -m "docs: update documentation"
git push origin main
```

GitHub Actions will automatically build and deploy to GitHub Pages.

**Option 2: Manual**

```bash
mkdocs gh-deploy
```

This builds and pushes to the `gh-pages` branch.

## 🔧 Configuration

### Update Repository URLs

Edit `mkdocs.yml` and replace placeholders:

```yaml
site_url: https://yourusername.github.io/ECGEN
repo_url: https://github.com/yourusername/ECGEN
```

Replace `yourusername` with your actual GitHub username.

### Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under **Source**, select:
   - Branch: `gh-pages`
   - Folder: `/ (root)`
4. Click **Save**
5. Your site will be available at `https://yourusername.github.io/ECGEN`

### Customize Theme

Edit `mkdocs.yml` to customize colors, features, etc:

```yaml
theme:
  name: material
  palette:
    - scheme: default
      primary: indigo      # Change this
      accent: indigo       # And this
```

Available colors: red, pink, purple, deep purple, indigo, blue, light blue, cyan, teal, green, light green, lime, yellow, amber, orange, deep orange

## 📝 Writing Documentation

### Add a New Page

1. **Create the file**:
```bash
touch docs/user-guide/new-page.md
```

2. **Add content**:
```markdown
# New Page Title

Your content here.

## Section 1

More content.
```

3. **Add to navigation** in `mkdocs.yml`:
```yaml
nav:
  - User Guide:
    - New Page: user-guide/new-page.md
```

4. **Preview**:
```bash
mkdocs serve
```

### Markdown Features

#### Code Blocks with Syntax Highlighting

````markdown
```python
from ecgen.models import Pulse2PulseGAN

model = Pulse2PulseGAN(config)
```
````

#### Admonitions (Callouts)

```markdown
!!! note
    This is a note.

!!! tip
    This is a helpful tip.

!!! warning
    This is a warning.

!!! danger
    This is dangerous!
```

#### Tabbed Content

```markdown
=== "Tab 1"
    Content for tab 1

=== "Tab 2"
    Content for tab 2
```

#### Tables

```markdown
| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
```

#### Links

```markdown
[Link text](other-page.md)
[External link](https://example.com)
```

## 🔍 API Documentation

API documentation is automatically generated from your Python docstrings.

### Write Good Docstrings

Use Google-style docstrings:

```python
def train_model(
    data_dir: str,
    batch_size: int = 128,
    max_epochs: int = 300,
) -> None:
    """Train the Pulse2Pulse model.
    
    This function trains a Pulse2Pulse WaveGAN model on the MIMIC-IV-ECG
    dataset with the specified hyperparameters.
    
    Args:
        data_dir: Path to MIMIC-IV-ECG dataset root directory
        batch_size: Number of samples per batch (default: 128)
        max_epochs: Maximum number of training epochs (default: 300)
        
    Returns:
        None. Model checkpoints are saved to the runs/ directory.
        
    Raises:
        ValueError: If data_dir does not exist
        RuntimeError: If CUDA is not available
        
    Examples:
        >>> train_model("/path/to/MIMIC-IV-ECG", batch_size=64)
        Training started...
        
    Note:
        This function requires a CUDA-capable GPU.
    """
    # Implementation
```

### API Docs Are Auto-Generated

The `docs/gen_ref_pages.py` script automatically:
1. Scans all Python files in `src/ecgen/`
2. Generates markdown pages with `mkdocstrings` directives
3. Creates navigation structure
4. Updates when you run `mkdocs serve` or `mkdocs build`

## 🎨 Customization

### Change Site Name

Edit `mkdocs.yml`:
```yaml
site_name: Your Project Name
```

### Add Logo

1. Add logo to `docs/assets/`:
```bash
mkdir -p docs/assets
cp logo.png docs/assets/
```

2. Update `mkdocs.yml`:
```yaml
theme:
  logo: assets/logo.png
```

### Add Favicon

```yaml
theme:
  favicon: assets/favicon.ico
```

### Add Social Links

```yaml
extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/yourusername
    - icon: fontawesome/brands/twitter
      link: https://twitter.com/yourusername
```

## 🐛 Troubleshooting

### Documentation Not Building

**Error**: `mkdocs build` fails

**Solution**:
```bash
# Reinstall dependencies
pip install -e ".[docs]" --force-reinstall

# Try building with verbose output
mkdocs build --verbose
```

### GitHub Pages Not Updating

**Problem**: Changes not visible on GitHub Pages

**Solutions**:
1. Check GitHub Actions: Repository → Actions tab
2. Look for failed workflows
3. Check `gh-pages` branch exists
4. Verify Pages settings (Settings → Pages)
5. Wait 5-10 minutes for propagation
6. Clear browser cache

### Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Use different port
mkdocs serve -a 127.0.0.1:8001

# Or kill existing process
lsof -ti:8000 | xargs kill -9
```

### API Documentation Empty

**Problem**: API reference pages are blank

**Solutions**:
1. Ensure docstrings exist in Python files
2. Check `docs/gen_ref_pages.py` is working
3. Rebuild from scratch:
```bash
rm -rf site/
mkdocs build
```

## 📚 File Structure

```
ECGEN/
├── docs/                              # Documentation source
│   ├── index.md                       # Homepage
│   ├── getting-started/               # Getting started guides
│   ├── user-guide/                    # User guides
│   ├── reference/                     # API reference
│   ├── development/                   # Development docs
│   ├── gen_ref_pages.py              # API doc generator
│   └── assets/                        # Images, etc.
├── mkdocs.yml                         # MkDocs configuration
├── .github/workflows/docs.yml         # Auto-deployment
├── docs_serve.sh                      # Convenience script
├── DOCS_MAINTENANCE.md               # Maintenance guide
└── DOCUMENTATION_SETUP.md            # This file
```

## 🔄 Workflow

### Daily Development

1. Edit documentation files in `docs/`
2. Run `mkdocs serve` to preview
3. Save changes (auto-reload)
4. Commit and push when satisfied

### Adding Features

1. Write code with docstrings
2. Add user guide if needed
3. Update changelog
4. Test locally
5. Push to trigger deployment

### Before Release

1. Update version in `mkdocs.yml`
2. Update changelog
3. Review all documentation
4. Test all examples
5. Deploy

## 📖 Resources

- **MkDocs Documentation**: https://www.mkdocs.org/
- **Material Theme**: https://squidfunk.github.io/mkdocs-material/
- **Mkdocstrings**: https://mkdocstrings.github.io/
- **Markdown Guide**: https://www.markdownguide.org/

## 💡 Tips

1. **Write as you code**: Update docs with code changes
2. **Use examples**: Show, don't just tell
3. **Keep it simple**: Clear language beats fancy words
4. **Test everything**: All code examples should work
5. **Link liberally**: Connect related documentation
6. **Mobile test**: Check on phone/tablet
7. **Search test**: Can users find what they need?

## 🎉 Next Steps

1. **Customize**: Update URLs and branding in `mkdocs.yml`
2. **Enable Pages**: Set up GitHub Pages in repository settings
3. **Write**: Add your project-specific documentation
4. **Deploy**: Push to main branch to trigger deployment
5. **Share**: Share your documentation URL!

## 📞 Support

- **Documentation Issues**: Check [DOCS_MAINTENANCE.md](DOCS_MAINTENANCE.md)
- **MkDocs Issues**: https://github.com/mkdocs/mkdocs/issues
- **Material Theme**: https://github.com/squidfunk/mkdocs-material/issues

---

**Ready to go?** Run `./docs_serve.sh` and start writing! 🚀
