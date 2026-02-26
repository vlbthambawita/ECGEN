# Documentation Quick Reference

Quick commands and tips for working with ECGEN documentation.

## 🚀 Essential Commands

### Installation
```bash
pip install -e ".[docs]"
```

### Local Development
```bash
# Start server (auto-reload on save)
mkdocs serve

# Or use convenience script
./docs_serve.sh

# Open http://127.0.0.1:8000
```

### Build & Deploy
```bash
# Build static site
mkdocs build

# Deploy to GitHub Pages (manual)
mkdocs gh-deploy

# Automatic deployment: just push to main/master
git push origin main
```

## 📁 File Locations

| What | Where |
|------|-------|
| Documentation source | `docs/` |
| Configuration | `mkdocs.yml` |
| API generator | `docs/gen_ref_pages.py` |
| GitHub Actions | `.github/workflows/docs.yml` |
| Setup guide | `DOCUMENTATION_SETUP.md` |
| Maintenance guide | `DOCS_MAINTENANCE.md` |
| Quick reference | `DOCS_QUICK_REFERENCE.md` (this file) |

## ✏️ Common Tasks

### Add a New Page
```bash
# 1. Create file
touch docs/user-guide/new-page.md

# 2. Add content (Markdown)
vim docs/user-guide/new-page.md

# 3. Add to navigation in mkdocs.yml
# 4. Preview
mkdocs serve
```

### Update Existing Page
```bash
# Edit and save - auto-reloads
vim docs/getting-started/quickstart.md
```

### Add Code Example
````markdown
```python
from ecgen.models import Pulse2PulseGAN
model = Pulse2PulseGAN(config)
```
````

### Add Callout
```markdown
!!! note
    Important information here

!!! tip
    Helpful tip here

!!! warning
    Warning message here
```

### Add Tabs
```markdown
=== "Option 1"
    Content for option 1

=== "Option 2"
    Content for option 2
```

## 🔧 Customization

### Update URLs (Important!)
Edit `mkdocs.yml`:
```yaml
site_url: https://YOUR_USERNAME.github.io/ECGEN
repo_url: https://github.com/YOUR_USERNAME/ECGEN
```

### Change Colors
```yaml
theme:
  palette:
    - scheme: default
      primary: indigo  # Change this
      accent: indigo   # And this
```

Colors: red, pink, purple, indigo, blue, cyan, teal, green, yellow, orange

### Add Logo
```yaml
theme:
  logo: assets/logo.png
```

## 📖 Documentation Structure

```
docs/
├── index.md                    # Homepage
├── getting-started/            # Installation & quickstart
├── user-guide/                 # Detailed guides
│   └── pulse2pulse/           # Pulse2Pulse specific
├── reference/                  # API docs (auto-generated)
└── development/                # Contributing & changelog
```

## 🎯 Best Practices

1. ✅ Write docs as you code
2. ✅ Include code examples
3. ✅ Test all examples
4. ✅ Use clear headings
5. ✅ Link related pages
6. ✅ Update changelog
7. ✅ Preview before committing

## 🐛 Troubleshooting

### Can't install dependencies?
```bash
pip install --upgrade pip
pip install -e ".[docs]" --force-reinstall
```

### Port already in use?
```bash
mkdocs serve -a 127.0.0.1:8001
```

### GitHub Pages not updating?
1. Check Actions tab
2. Wait 5-10 minutes
3. Clear browser cache
4. Check Settings → Pages

### Build errors?
```bash
mkdocs build --verbose
```

## 📚 Markdown Syntax

### Headings
```markdown
# H1
## H2
### H3
```

### Links
```markdown
[Text](page.md)
[External](https://example.com)
```

### Images
```markdown
![Alt text](assets/image.png)
```

### Tables
```markdown
| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
```

### Lists
```markdown
- Item 1
- Item 2
  - Nested item

1. First
2. Second
```

## 🔍 API Documentation

### Write Good Docstrings
```python
def function(param: str) -> bool:
    """Brief description.
    
    Longer description.
    
    Args:
        param: Parameter description
        
    Returns:
        Return value description
        
    Examples:
        >>> function("test")
        True
    """
    return True
```

API docs auto-generate from docstrings!

## 📞 Help & Resources

| Resource | Link |
|----------|------|
| Setup Guide | [DOCUMENTATION_SETUP.md](DOCUMENTATION_SETUP.md) |
| Maintenance | [DOCS_MAINTENANCE.md](DOCS_MAINTENANCE.md) |
| Summary | [DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md) |
| MkDocs | https://www.mkdocs.org/ |
| Material Theme | https://squidfunk.github.io/mkdocs-material/ |

## ⚡ Quick Checklist

Before first deployment:
- [ ] Install dependencies: `pip install -e ".[docs]"`
- [ ] Update URLs in `mkdocs.yml`
- [ ] Test locally: `mkdocs serve`
- [ ] Enable GitHub Pages (Settings → Pages)
- [ ] Push to main/master branch
- [ ] Wait 2-3 minutes
- [ ] Visit your docs URL

## 🎉 That's It!

You're ready to maintain professional documentation for ECGEN.

**Most common workflow:**
1. Edit `.md` files in `docs/`
2. Run `mkdocs serve` to preview
3. Commit and push
4. Documentation auto-deploys!

---

**Need more details?** See [DOCUMENTATION_SETUP.md](DOCUMENTATION_SETUP.md)
