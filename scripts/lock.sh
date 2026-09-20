#!/usr/bin/env sh
set -eu
uv lock
uv export --format pylock.toml --output-file pylock.toml
# Optional SBOM for release provenance; uv currently marks CycloneDX export as preview.
uv export --format cyclonedx1.5 --output-file sbom.cdx.json
