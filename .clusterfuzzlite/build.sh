#!/bin/bash -eu
# Build the Jazzer.js fuzz targets for ClusterFuzzLite / OSS-Fuzz.
#
# byom sits between untrusted parties on both sides of the wire: clients hand
# it arbitrary Anthropic-shaped /v1/messages bodies, and the upstream hands
# back OpenAI-shaped responses and SSE streams that byom parses, translates,
# and rewrites in both directions. Each target feeds hostile bytes into one of
# those translators and asserts its fail-safe contract — see the header of each
# fuzz/*.fuzz.js.

cd "$SRC/byom"

# npm ci verifies every integrity hash in the committed lockfile
# (Scorecard Pinned-Dependencies). byom has no runtime deps; this installs the
# devDependencies used by the build.
npm ci --no-audit --no-fund

# Jazzer.js is installed build-side rather than as a devDependency so the
# published package's dependency tree stays exactly as committed (byom
# advertises itself as dependency-free). It comes from its own lockfile
# (.clusterfuzzlite/package-lock.json) so every byte is integrity-checked, then
# gets merged into the project node_modules where compile_javascript_fuzzer
# expects to resolve it.
npm ci --prefix .clusterfuzzlite --no-audit --no-fund
cp -r .clusterfuzzlite/node_modules/. node_modules/

# The fuzz targets exercise the compiled output (dist/), same as the test
# suite — build it first.
npm run build

for target in parse_sse_line openai_to_anthropic sse_translate; do
  compile_javascript_fuzzer byom "fuzz/${target}.fuzz.js" --sync
done
