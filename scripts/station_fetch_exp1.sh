#!/usr/bin/env bash
# Run ON the Station (aarch64), detached. Downloads the EXP-1 stage in two parallel
# lanes (the 64.6 GB index alone; everything else), verifies sha256 against
# manifest.tsv, then concatenates the index and gunzips the corpus.
#   setsid -f nohup bash scripts/station_fetch_exp1.sh > /models/treeopsd/fetch.log 2>&1 < /dev/null
set -uo pipefail
ROOT="${TREEOPSD_DATA_ROOT:-/models/treeopsd}"
cd "$(dirname "$0")/.."
echo "[$(date -Is)] start root=$ROOT pid=$$"
python3 scripts/fetch.py --root "$ROOT" --yes --groups wiki18-index > "$ROOT/fetch.index.log" 2>&1 &
P1=$!
python3 scripts/fetch.py --root "$ROOT" --yes \
  --groups mhqa,model-e5-base-v2,model-qwen3-1.7b,model-qwen3-4b-thinking,wiki18 > "$ROOT/fetch.rest.log" 2>&1 &
P2=$!
echo "lane pids: index=$P1 rest=$P2"
wait $P2; R2=$?; echo "[$(date -Is)] rest lane exit=$R2"; cat "$ROOT/fetch.rest.log"
if [[ $R2 -eq 0 && ! -f "$ROOT/wiki18/wiki-18.jsonl" ]]; then
  tar -xzOf "$ROOT/wiki18/wiki-18.jsonl.gz" data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl > "$ROOT/wiki18/wiki-18.jsonl.tmp" && mv "$ROOT/wiki18/wiki-18.jsonl.tmp" "$ROOT/wiki18/wiki-18.jsonl"
  echo "[$(date -Is)] corpus lines: $(wc -l < "$ROOT/wiki18/wiki-18.jsonl")"
fi
# Tier-2 headline eval data (BrowseComp-Plus slice + Qwen3-Embedding-8B), after the EXP-1 set
python3 scripts/fetch.py --root "$ROOT" --yes \
  --groups bcplus,bcplus-corpus,bcplus-index-q3e8b,bcplus-index-bm25,model-qwen3-embedding-8b,or-questions > "$ROOT/fetch.bcplus.log" 2>&1
R3=$?; echo "[$(date -Is)] bcplus lane exit=$R3"; cat "$ROOT/fetch.bcplus.log"
wait $P1; R1=$?; echo "[$(date -Is)] index lane exit=$R1"; cat "$ROOT/fetch.index.log"
IDX="$ROOT/wiki18-index"
if [[ $R1 -eq 0 && ! -f "$IDX/e5_Flat.index" ]]; then
  cat "$IDX/part_aa" "$IDX/part_ab" > "$IDX/e5_Flat.index.tmp" && mv "$IDX/e5_Flat.index.tmp" "$IDX/e5_Flat.index"
fi
if [[ -f "$IDX/e5_Flat.index" ]]; then
  SZ=$(stat -c %s "$IDX/e5_Flat.index"); echo "e5_Flat.index bytes=$SZ (expect 64559075373)"
  [[ "$SZ" == 64559075373 ]] && sha256sum "$IDX/e5_Flat.index" | tee "$IDX/e5_Flat.index.sha256"
fi
for f in hotpotqa/train.jsonl hotpotqa/dev.jsonl 2wikimultihopqa/dev.jsonl musique/dev.jsonl bamboogle/test.jsonl; do
  [[ -f "$ROOT/mhqa/$f" ]] && echo "rows $f: $(wc -l < "$ROOT/mhqa/$f")"
done
echo "[$(date -Is)] DONE rest=$R2 index=$R1 bcplus=$R3"
