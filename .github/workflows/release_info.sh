#!/bin/bash

set -e

# Get latest version tag
get_last_tag() {
  curl --silent "https://api.github.com/repos/fishjojo/pyscfad/releases/latest" | sed -n 's/.*"tag_name": "v\(.*\)",.*/\1/p'
}
last_version=$(get_last_tag)
echo Last version: $last_version

# Get current version tag
cur_version=$(sed -n "/^__version__ =/s/.*\"\(.*\)\"/\1/p" pyscfad/version.py)
if [ -z "$cur_version" ]; then
  cur_version=$(sed -n "/^__version__ =/s/.*'\(.*\)'/\1/p" pyscfad/version.py)
fi
echo Current version: $cur_version

# Create version tag
if [ -n "$last_version" ] && [ -n "$cur_version" ] && [ "$cur_version" != "$last_version" ]; then
  git config user.name "Github Actions"
  git config user.email "github-actions@users.noreply.github.com"
  version_tag=v"$cur_version"

  # Extract release info from CHANGELOG
  sed -n "/^## pyscfad $cur_version/,/^## pyscfad $last_version/p" CHANGELOG.md | tail -n +2 | sed -e '/^## pyscfad /,$d' | head -n -1 > RELEASE.md
  echo "::set-output name=version_tag::$version_tag"
fi
