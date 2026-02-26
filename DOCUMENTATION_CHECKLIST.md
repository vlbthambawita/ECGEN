# Documentation Setup Checklist

Use this checklist to complete your documentation setup and deployment.

## ✅ Pre-Deployment Checklist

### 1. Install Dependencies
- [ ] Run `pip install -e ".[docs]"`
- [ ] Verify installation: `mkdocs --version`

### 2. Test Locally
- [ ] Run `mkdocs serve` or `./docs_serve.sh`
- [ ] Open http://127.0.0.1:8000 in browser
- [ ] Check all pages load correctly
- [ ] Test navigation menu
- [ ] Test search functionality
- [ ] Verify code examples display correctly
- [ ] Check mobile view (resize browser)

### 3. Customize Configuration
- [ ] Open `mkdocs.yml`
- [ ] Replace `yourusername` with your GitHub username in:
  - `site_url: https://yourusername.github.io/ECGEN`
  - `repo_url: https://github.com/yourusername/ECGEN`
  - `repo_name: yourusername/ECGEN`
- [ ] Update `site_author` if desired
- [ ] Customize colors (optional):
  - `primary: indigo` → your choice
  - `accent: indigo` → your choice
- [ ] Save changes

### 4. Update Documentation Content
- [ ] Review `docs/index.md` (homepage)
- [ ] Update installation paths in `docs/getting-started/installation.md`
- [ ] Verify all examples in quick start work
- [ ] Add any project-specific information
- [ ] Update social links in `mkdocs.yml` (optional)

### 5. Prepare for Deployment
- [ ] Ensure you have a GitHub repository
- [ ] Verify `.github/workflows/docs.yml` exists
- [ ] Check `.gitignore` includes documentation artifacts
- [ ] Commit all changes:
  ```bash
  git add .
  git commit -m "docs: add MkDocs documentation system"
  ```

## 🚀 Deployment Checklist

### 6. Enable GitHub Pages
- [ ] Go to your repository on GitHub
- [ ] Click **Settings** tab
- [ ] Scroll to **Pages** section (left sidebar)
- [ ] Under **Source**, select:
  - Branch: `gh-pages` (will be created automatically)
  - Folder: `/ (root)`
- [ ] Click **Save**
- [ ] Note: You may see a message that `gh-pages` doesn't exist yet - this is normal

### 7. First Deployment
- [ ] Push to main/master branch:
  ```bash
  git push origin main
  ```
- [ ] Go to **Actions** tab on GitHub
- [ ] Watch the "Deploy Documentation" workflow run
- [ ] Wait for workflow to complete (green checkmark)
- [ ] Verify `gh-pages` branch was created (Branches tab)

### 8. Verify Deployment
- [ ] Wait 2-3 minutes for GitHub Pages to update
- [ ] Visit `https://yourusername.github.io/ECGEN` (replace yourusername)
- [ ] Verify homepage loads
- [ ] Test navigation
- [ ] Check all pages
- [ ] Test search
- [ ] Verify API reference generated correctly

## 📝 Post-Deployment Checklist

### 9. Update Repository README
- [ ] Add documentation badge (optional):
  ```markdown
  [![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://yourusername.github.io/ECGEN)
  ```
- [ ] Add link to documentation in README
- [ ] Commit and push changes

### 10. Set Up Custom Domain (Optional)
- [ ] Purchase/configure domain
- [ ] Add `CNAME` file to `docs/`:
  ```bash
  echo "docs.yourdomain.com" > docs/CNAME
  ```
- [ ] Configure DNS:
  ```
  CNAME docs.yourdomain.com -> yourusername.github.io
  ```
- [ ] Update `site_url` in `mkdocs.yml`
- [ ] Commit, push, and wait for deployment

### 11. Ongoing Maintenance
- [ ] Read `DOCS_MAINTENANCE.md`
- [ ] Bookmark `DOCS_QUICK_REFERENCE.md`
- [ ] Set up documentation update workflow:
  - Write code → Update docstrings
  - Add features → Update user guide
  - Fix bugs → Update changelog
  - Push → Auto-deploy

## 🐛 Troubleshooting

### If GitHub Actions Fails
- [ ] Check Actions tab for error message
- [ ] Verify `pyproject.toml` has `[project.optional-dependencies]` docs section
- [ ] Ensure all documentation files are committed
- [ ] Check `mkdocs.yml` syntax
- [ ] Try building locally: `mkdocs build`

### If GitHub Pages Shows 404
- [ ] Verify `gh-pages` branch exists
- [ ] Check Pages settings (Settings → Pages)
- [ ] Wait 5-10 minutes (can take time to propagate)
- [ ] Clear browser cache
- [ ] Try incognito/private browsing mode

### If Local Server Won't Start
- [ ] Check port 8000 is not in use
- [ ] Try different port: `mkdocs serve -a 127.0.0.1:8001`
- [ ] Reinstall dependencies: `pip install -e ".[docs]" --force-reinstall`
- [ ] Check for Python errors in terminal

### If API Reference Is Empty
- [ ] Ensure Python files have docstrings
- [ ] Check `docs/gen_ref_pages.py` exists
- [ ] Verify `src/ecgen/` directory structure
- [ ] Rebuild: `mkdocs build --clean`

## ✨ Optional Enhancements

### Add Logo
- [ ] Create/obtain logo image (PNG recommended)
- [ ] Save to `docs/assets/logo.png`
- [ ] Update `mkdocs.yml`:
  ```yaml
  theme:
    logo: assets/logo.png
  ```

### Add Favicon
- [ ] Create favicon.ico
- [ ] Save to `docs/assets/favicon.ico`
- [ ] Update `mkdocs.yml`:
  ```yaml
  theme:
    favicon: assets/favicon.ico
  ```

### Add More Social Links
- [ ] Update `extra` section in `mkdocs.yml`:
  ```yaml
  extra:
    social:
      - icon: fontawesome/brands/github
        link: https://github.com/yourusername
      - icon: fontawesome/brands/twitter
        link: https://twitter.com/yourusername
      - icon: fontawesome/brands/linkedin
        link: https://linkedin.com/in/yourusername
  ```

### Enable Google Analytics (Optional)
- [ ] Get Google Analytics ID
- [ ] Add to `mkdocs.yml`:
  ```yaml
  extra:
    analytics:
      provider: google
      property: G-XXXXXXXXXX
  ```

## 📊 Success Criteria

Your documentation setup is complete when:

- ✅ Local server runs without errors
- ✅ All pages load and display correctly
- ✅ Navigation works properly
- ✅ Search returns results
- ✅ Code blocks have syntax highlighting
- ✅ API reference is generated
- ✅ GitHub Actions workflow succeeds
- ✅ GitHub Pages site is accessible
- ✅ Documentation updates automatically on push

## 📚 Reference Documents

- **Setup Guide**: [DOCUMENTATION_SETUP.md](DOCUMENTATION_SETUP.md)
- **Maintenance Guide**: [DOCS_MAINTENANCE.md](DOCS_MAINTENANCE.md)
- **Quick Reference**: [DOCS_QUICK_REFERENCE.md](DOCS_QUICK_REFERENCE.md)
- **Summary**: [DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md)

## 🎉 Completion

Once all items are checked:

- [ ] Documentation is fully functional
- [ ] Auto-deployment is working
- [ ] Team members can access online docs
- [ ] You know how to maintain and update docs

**Congratulations! Your documentation system is ready!** 🎊

---

**Need help?** Refer to the guides above or check:
- MkDocs: https://www.mkdocs.org/
- Material Theme: https://squidfunk.github.io/mkdocs-material/
