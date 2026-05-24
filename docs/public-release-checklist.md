# Public release checklist

Steps taken before making a private repo public. Reuse for other projects.

## 1. Audit for sensitive content

```bash
# Personal paths
grep -r "/Users/<name>" --include="*.py" --include="*.sh" --include="*.plist" --include="*.md" .

# Credentials and tokens
grep -rEi "(api_key|secret|password|token|bearer)\s*=" --include="*.py" --include="*.js" --include="*.env" .

# Machine identifiers (Tailscale, hostnames)
grep -r "\.ts\.net\|tailscale\|<machine-name>" .

# Database files with real data
find . -name "*.db" -o -name "*.db-shm" -o -name "*.db-wal"
```

## 2. Fix .gitignore

Add any file that contains local paths, machine-specific config, or real data:

```
*.db-shm
*.db-wal
<config-file-with-paths>     # add the real file
<config-file-with-paths>.template  # commit only the template
```

## 3. Replace sensitive files with templates

For any committed file with hardcoded local paths or identifiers:

1. Create `<filename>.template` with `<PLACEHOLDER>` substitutions
2. Add the original filename to `.gitignore`
3. Add setup instructions to README

## 4. Fix documentation

- Replace old repo name references (`grep -r "old-repo-name" --include="*.md" .`)
- Update any titles or headers that refer to internal project names
- Add a section to README explaining how to fill in the template files

## 5. Rewrite git history

If sensitive files were committed, remove them from all commits:

```bash
# Stash working changes first
git stash push -u -m "pre-filter stash"

# Remove files from all commits
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch <file1> <file2>' \
  --prune-empty --tag-name-filter cat -- --all

# Clean up
rm -rf .git/refs/original
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Restore working changes
git stash pop
```

Then force-push: `git push origin main --force`

## 6. Final verification

```bash
# Confirm nothing sensitive remains in tracked files
git grep -i "<sensitive-term>"

# Confirm files are gone from history
git log --all --oneline -- <removed-file>   # should produce no output
```
