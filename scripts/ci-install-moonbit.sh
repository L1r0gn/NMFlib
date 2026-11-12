#!/usr/bin/env bash
# Install the MoonBit toolchain with the official installer, then fail closed
# if the resolved version differs from the locally verified pin.
#
# Why not install a pinned version directly? The official CDN
# (cli.moonbitlang.com) only publishes the `latest` and `nightly` channels;
# date-pinned artifact keys return 403 (verified 2026-11-12). To avoid
# silently testing against an unverified `latest`, CI installs the current
# stable and asserts the exact version below. When MoonBit publishes a new
# stable, this check fails loudly: verify the new toolchain locally
# (moon check --deny-warn + moon test on every backend), then update
# EXPECTED_MOON_VERSION in .github/workflows/ci.yml.
set -euo pipefail

EXPECTED_MOON_VERSION="${EXPECTED_MOON_VERSION:?EXPECTED_MOON_VERSION must be set}"

curl -fsSL https://cli.moonbitlang.com/install/unix.sh | bash
echo "$HOME/.moon/bin" >> "$GITHUB_PATH"
export PATH="$HOME/.moon/bin:$PATH"

version_output="$(moon version)"
printf '%s\n' "$version_output"

actual="$(printf '%s\n' "$version_output" | head -n 1 | awk '{print $2}')"
if [ "$actual" != "$EXPECTED_MOON_VERSION" ]; then
  echo "::error::MoonBit toolchain drift: expected ${EXPECTED_MOON_VERSION}, got ${actual}."
  echo "Verify the new toolchain locally, then update EXPECTED_MOON_VERSION in .github/workflows/ci.yml."
  exit 1
fi
echo "MoonBit ${actual} matches the verified pin."