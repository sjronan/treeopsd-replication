#!/usr/bin/env bash
# Stage the frozen retrieval corpora, indexes, datasets and models for the
# TreeOPSD replication. DRY RUN unless YES=1. Nothing multi-GB is fetched by
# default. All files are pinned by revision and verified by sha256 (scripts/manifest.tsv).
#
#   TREEOPSD_DATA_ROOT=/data/treeopsd STAGE=exp1 ./scripts/fetch_data.sh          # dry run
#   TREEOPSD_DATA_ROOT=/data/treeopsd STAGE=exp1 YES=1 ./scripts/fetch_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TREEOPSD_DATA_ROOT:=./data}"
: "${STAGE:=exp1}"
YESFLAG=""; [[ "${YES:-0}" == "1" ]] && YESFLAG="--yes"

case "$STAGE" in
  shakedown) SEL="mhqa,model-qwen3-1.7b,model-e5-base-v2,wiki18,wiki18-index" ;;     # ~74.9 GB
  exp1)      SEL="mhqa,model-qwen3-4b-thinking,model-qwen3-1.7b,model-e5-base-v2,wiki18,wiki18-index" ;;  # ~82.9 GB
  bcplus)    SEL="bcplus,bcplus-corpus,bcplus-index-q3e8b,bcplus-index-bm25,model-qwen3-embedding-8b,or-questions" ;;  # ~24.0 GB
  or-train)  SEL="or-corpus" ;;                                                          # ~29.8 GB (BM25 built locally)
  or-dense)  SEL="or-index-q3e8b" ;;                                                     # ~122 GB pickle, see DATA.md
  scale30b)  SEL="model-qwen3-30b-a3b-thinking" ;;                                       # ~61.1 GB
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
python3 scripts/fetch.py --groups "$SEL" --root "$TREEOPSD_DATA_ROOT" $YESFLAG

# Post-processing (only after a real download)
if [[ -n "$YESFLAG" && ",$SEL," == *",wiki18-index,"* ]]; then
  IDX="$TREEOPSD_DATA_ROOT/wiki18-index"
  if [[ ! -f "$IDX/e5_Flat.index" ]]; then
    cat "$IDX/part_aa" "$IDX/part_ab" > "$IDX/e5_Flat.index.tmp" && mv "$IDX/e5_Flat.index.tmp" "$IDX/e5_Flat.index"
  fi
  test "$(stat -c %s "$IDX/e5_Flat.index")" -eq 64559075373 || { echo "e5_Flat.index size wrong" >&2; exit 3; }
  sha256sum "$IDX/e5_Flat.index" | tee "$IDX/e5_Flat.index.sha256"   # no published hash; record ours
  echo "parts verified individually; delete part_aa/part_ab manually after recording the hash"
fi
if [[ -n "$YESFLAG" && ",$SEL," == *",wiki18,"* ]]; then
  C="$TREEOPSD_DATA_ROOT/wiki18"
  # NOTE: despite its name the .gz is a gzipped TAR holding one member (wiki_dump.jsonl)
  [[ -f "$C/wiki-18.jsonl" ]] || tar -xzOf "$C/wiki-18.jsonl.gz" data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl > "$C/wiki-18.jsonl"
fi
