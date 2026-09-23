#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p graphs
# Official archive linked in the upstream README; extract only Reddit.
curl -fL --retry 3 'https://drive.usercontent.google.com/download?id=1_sE65oveGpzRdCcExBmUaNG982lUB-Cx&export=download&confirm=t' |
  tar -xz --wildcards --strip-components=1 -C graphs '18graphs/reddit.dgl.*'
