# ECGEN Documentation

This directory contains the source files for the ECGEN documentation.

## Quick Start

### Install Dependencies

```bash
pip install -e ".[docs]"
```

### Serve Locally

```bash
mkdocs serve
```

Open http://127.0.0.1:8000 in your browser.

### Build

```bash
mkdocs build
```

Output will be in the `site/` directory.

### Deploy

```bash
mkdocs gh-deploy
```

Or push to main/master branch for automatic deployment via GitHub Actions.

## Structure

- `index.md` - Homepage
- `getting-started/` - Installation and quick start guides
- `user-guide/` - Detailed user guides
- `reference/` - API reference (auto-generated)
- `development/` - Contributing and changelog
- `gen_ref_pages.py` - Script to generate API docs

## Maintenance

See [DOCS_MAINTENANCE.md](../DOCS_MAINTENANCE.md) for detailed maintenance instructions.

## Contributing

When adding documentation:

1. Create/edit Markdown files in appropriate directory
2. Update navigation in `mkdocs.yml`
3. Test locally with `mkdocs serve`
4. Commit and push changes

See [Contributing Guide](development/contributing.md) for more details.
