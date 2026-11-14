#!/usr/bin/env python3
"""Audit every numeric occurrence in the abstract, Results and Mechanism sections.

No TeX edits, network, GPU, or third-party dependencies. Run from any directory.
JSON statistics take precedence over the analysis prose. Bootstrap endpoints are
read from the released packages, not independently re-bootstrapped (question-level
resampling inputs are not among these three sources). Analysis-only telemetry is
corroboration, not an independent recomputation. Unmapped claims fail closed.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from zoneinfo import ZoneInfo
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PAPER = 'paper/arxiv-package/main.tex'
ANALYSIS = 'ANALYSIS-FINAL-20261024.md'
CB = 'results/seed_package_CB.json'
CE = 'results/seed_package_CE_gb300.json'
PACKAGES = {p: json.loads((ROOT / p).read_text()) for p in (CB, CE)}
analysis = (ROOT / ANALYSIS).read_text()
# Keep original offsets while normalizing TeX thousands separators and signs.
raw = (ROOT / PAPER).read_text()
text = raw.replace('{,}', ',  ').replace('{+}', ' + ')
NUM = re.compile(r'(?<![\d.])[-+]?\d+(?:, {0,2}\d{3})*(?:\.\d+)?|\b(?:zero|one|two|three|four|five|sevenfold)\b', re.I)
WORDS = dict(zero=0, one=1, two=2, three=3, four=4, five=5, sevenfold=7)
rows = []
assigned = {}


def source(path, key, value, method='source transcription'):
    return dict(source_path=path, source_key=key, source_value=value,
                recomputed_value=value, method=method)


def js(path, *keys):
    value = PACKAGES[path]
    for k in keys:
        value = value[k]
    return source(path, ''.join(f'[{json.dumps(k)}]' for k in keys), value)


def calc(path, key, vals, fn, formula):
    r = source(path, key, vals, formula)
    r['recomputed_value'] = fn(vals)
    return r


def ev(start, end=None):
    a = analysis.index(start)
    b = analysis.index(end, a) + len(end) if end else analysis.index('\n\n', a) if '\n\n' in analysis[a:] else len(analysis)
    return a, b


def analysis_value(token, spans):
    val = number(token)
    for a, b in spans:
        for m in NUM.finditer(analysis, a, b):
            if number(m.group()) == val:
                line = analysis.count('\n', 0, m.start()) + 1
                return source(ANALYSIS, f'line {line}: {analysis.splitlines()[line-1]}', val,
                              'analysis corroboration only; raw telemetry not supplied')
    return None


def number(t):
    return WORDS[t.lower()] if t.lower() in WORDS else float(t.replace(",", "").replace(" ", ""))


def tokens(a, b):
    return list(NUM.finditer(text, a, b))


def put(m, ref):
    assigned[m.start()] = ref


def sequence(fragment, refs, start=0):
    a = text.index(fragment, start)
    ms = tokens(a, a + len(fragment))
    if len(ms) != len(refs):
        raise ValueError((fragment, [m.group() for m in ms], len(refs)))
    for m, r in zip(ms, refs):
        put(m, r)


def stats(path, pair, metric):
    values = [PACKAGES[path]['pairs'][f'{pair} s{i}'][metric][0] for i in range(3)]
    # Recompute using the exact two-sided 95% critical value for df=2.
    return [calc(path, f'pairs.{pair} s[0:3].{metric}[0]', values,
                 lambda v, sign=sign: statistics.mean(v) + sign * 4.302652729911275 * statistics.stdev(v) / math.sqrt(3),
                 f'mean {sign:+} t(0.975, df=2)=4.302652729911275 * sample_sd / sqrt(3)') for sign in (-1, 1)]


def pooled(path, metric):
    return [js(path, 'pooled', metric, i) for i in range(3)]


def seeds(path, pair, metric):
    return [js(path, 'pairs', f'{pair} s{i}', metric, 0) for i in range(3)]


def corroborate(a, b, spans):
    for m in tokens(a, b):
        if m.start() not in assigned:
            r = analysis_value(m.group(), spans)
            if r:
                put(m, r)


abstract_start = text.index('\\begin{abstract}')
abstract_end = text.index('\\end{abstract}')
results_start = text.index('\\section{Results}')
results_end = text.index('\\section{Discussion')
# Metric cells use named JSON keys, never a search for the same numeric value.
tables = list(re.finditer(r'\\begin\{table\}.*?\\end\{table\}', text[results_start:results_end], re.S))
t1start = results_start + tables[0].start()
t1 = tables[0].group()
header_line = next(line for line in t1.splitlines() if line.startswith('Run &'))
column_keys = {'BC+':'bcplus', 'HotpotQA':'hotpotqa', '2Wiki':'2wikimultihopqa', 'MuSiQue':'musique', 'Bamboogle':'bamboogle', 'QA avg':'qa_avg'}
metrics = [column_keys[re.sub(r'\\+.*$', '', cell).strip()] for cell in header_line.split('&')[1:]]
for line in t1.splitlines():
    if not re.match(r'^[ABCDE] ', line):
        continue
    arm = line[0]
    cells = line.split('&')[1:]
    line_start = text.index(line, t1start)
    for cell, metric in zip(cells, metrics):
        pos = text.index(cell, line_start)
        for i, m in enumerate(tokens(pos, pos+len(cell))):
            path = CE if arm in 'CE' else CB
            ref = js(path, 'runs', f'{arm}s{i}', metric)
            if metric == 'qa_avg':
                vals = [PACKAGES[path]['runs'][f'{arm}s{i}'][k] for k in ['hotpotqa','2wikimultihopqa','musique','bamboogle']]
                ref['recomputed_value'] = statistics.mean(vals)
                ref['method'] = 'unweighted mean of four benchmark EM values; source_value is stored qa_avg'
            put(m, ref)

t2start = results_start + tables[1].start()
for line in tables[1].group().splitlines():
    if not line.startswith('C$-$'):
        continue
    pair = re.match(r'C\$-\$([EBAD])', line).group(1)
    cells = line.split('&')
    metrics = ['tier1_qa', 'tier2_bcplus'] if pair in 'AD' else ['tier1_qa' if cells[1].strip() == 'QA' else 'tier2_bcplus']
    path = CB if pair == 'B' else CE
    refs = []
    for metric in metrics:
        refs += [js(path, 'pairs', f'C-{pair} s0', metric, i) for i in range(3)] if pair in 'AD' else pooled(path, metric)
    frag = '&'.join(cells[2:])
    if pair in 'BE':
        refs += seeds(path, f'C-{pair}', metrics[0]) + stats(path, f'C-{pair}', metrics[0])
    sequence(frag, refs, t2start)

# Repeated headline statistics in abstract and results prose.
for a,b in [(abstract_start,abstract_end),(results_start,t1start),(results_start+tables[0].end(),t2start)]:
    part=text[a:b]
    for metric, trio in [('tier1_qa',r'\+3\.76.*?\[\+2\.81, \+4\.7\d\]'),('tier2_bcplus',r'\+3\.45.*?\[\+2\.\d\d, \+4\.78\]')]:
        for m in re.finditer(trio,part,re.S):sequence(m.group(),pooled(CB,metric),a+m.start())
    for metric,pat in [('tier1_qa',r'\+1\.98.*?\[\+1\.03, \+2\.94\]'),('tier2_bcplus',r'\+0\.08.*?\[-1\.24, \+1\.41\]')]:
        for m in re.finditer(pat,part,re.S):sequence(m.group(),pooled(CE,metric),a+m.start())
    for m in re.finditer(r'3\.78, 2\.10, 0\.05',part):sequence(m.group(),seeds(CE,'C-E','tier1_qa'),a+m.start())
    for metric,pat in [('tier1_qa',r'\[-2\.66, \+6\.61\]'),('tier1_qa',r'\[-4\.06, \+11\.57\]'),('tier2_bcplus',r'\[-6\.13, \+13\.03\]')]:
        for m in re.finditer(pat,part):sequence(m.group(),stats(CE if number(tokens(a+m.start(),a+m.end())[0].group()) == -2.66 else CB,'C-E' if number(tokens(a+m.start(),a+m.end())[0].group()) == -2.66 else 'C-B',metric),a+m.start())
    for m in re.finditer(r'-1\.81, -3\.01, \+5\.06',part):sequence(m.group(),seeds(CE,'C-E','tier2_bcplus'),a+m.start())
    for m in re.finditer(r'\+1\.81, \+4\.82, \+2\.29',part):sequence(m.group(),seeds(CE,'E-B','tier2_bcplus'),a+m.start())
# Caption hardware-matched per-seed C-B (not pooled).
for fragment,metric in [('4.41, -0.18, 6.47','tier1_qa'),('0.00, 1.81, 7.35','tier2_bcplus')]:
    sequence(fragment,seeds(CE,'C-B',metric),t2start)
# Excluding seed 2: recompute from as-scored CB pair values.
for fragment,metric in [('+2.40','tier1_qa'),('+1.51','tier2_bcplus')]:
    vals=[PACKAGES[CB]['pairs'][f'C-B s{i}'][metric][0] for i in range(2)]
    sequence(fragment,[calc(CB,f'pairs.C-B s[0:2].{metric}[0]',vals,statistics.mean,'mean of seed 0 and seed 1 gaps')],results_start)
# Explicit arm seed SDs, including abstract.
for m in re.finditer(r'0\.80',text[abstract_start:results_end]):
    pos=abstract_start+m.start()
    if not (pos<abstract_end or pos>=results_start):continue
    vals=[PACKAGES[CE]['runs'][f'Es{i}']['qa_avg'] for i in range(3)]
    put(tokens(pos,pos+4)[0],calc(CE,'runs.Es[0:3].qa_avg',vals,statistics.stdev,'sample standard deviation across seeds'))
for literal,arm,path in [('1.07','C',CE),('3.95','B',CB)]:
    pos=text.index(literal,results_start);vals=[PACKAGES[path]['runs'][f'{arm}s{i}']['qa_avg'] for i in range(3)]
    put(tokens(pos,pos+4)[0],calc(path,f'runs.{arm}s[0:3].qa_avg',vals,statistics.stdev,'sample standard deviation across seeds'))

# The remaining telemetry, design counts and identifiers are checked only in
# semantically scoped analysis passages. A same-valued number elsewhere cannot
# validate them. Absent source evidence remains MISMATCH.
header=[ev('Study:', 'Seed-level t-intervals below are n=3, t(2)=4.30,')]
ce_spans=[ev('Hardware-matched (GB300)', '## 2.'),ev('Per-arm step-100 means', 'Limitations to print')]
cb_spans=[ev('## 2.', '## 3.')]
mech=[ev('## 3.', '## 4.')]
budget=[ev('## 4.', '## 5.')]
selection=[ev('## 5.', '## 6.')]
verdict=[ev('## 6.', 'Per-arm step-100 means')]
# Abstract: scope sentence by topic (rather than matching all analysis numbers).
for m in re.finditer(r'[^.!?]*(?:\.(?!\d)|[!?]|$)',text[abstract_start:abstract_end]):
    a=abstract_start+m.start();b=abstract_start+m.end();s=m.group()
    spans = header
    if 'Against the paper' in s or 'passes are fragile' in s:spans=cb_spans+header
    if 'added control' in s:spans=ce_spans+header
    if 'posterior term' in s or 'dose' in s:spans=mech+verdict
    if 'budget match' in s or 'trained tokens' in s:spans=budget
    if 'boundedness' in s:spans=ce_spans+verdict
    corroborate(a,b,spans)
corroborate(abstract_start,abstract_end,header+cb_spans+ce_spans+mech+budget+verdict)
# Each results paragraph has a separate evidence scope.
for a,b,spans in [
    (t1start,results_start+tables[0].end(),header+cb_spans+verdict+mech),
    (results_start+tables[0].end(),text.index("\\paragraph{Against",results_start),ce_spans+header),
    (text.index("\\paragraph{Against",results_start),t2start,cb_spans+header),
    (t2start,results_start+tables[1].end(),header+ce_spans+cb_spans),
    (text.index('\\section{Mechanism'),text.index('\\paragraph{Budget'),mech+verdict+header),
    (text.index('\\paragraph{Budget'),text.index('\\paragraph{Selection'),budget),
    (text.index('\\paragraph{Selection'),results_end,selection+header),
]:corroborate(a,b,spans)
# Explicit analysis evidence for design labels, table confidence and known
# abstract paraphrases whose numeric form is spelled out differently upstream.
for a,b in [(abstract_start,abstract_end),(results_start,results_end)]:
    for m in tokens(a,b):
        if m.start() in assigned:continue
        context=text[max(a,m.start()-24):min(b,m.end()+24)]
        if m.group().lower()=='sevenfold':
            vals=[4.79,1.05];ref=calc(ANALYSIS,'§3 E=16 full-run median drift / E=4 full-run median drift',vals,lambda v:v[0]/v[1],'4.79 / 1.05 = 4.5619; same-window median ratio. Sevenfold uses mixed time windows.')
            put(m,ref)
        elif m.group()=='95' and 'CI' in context:
            put(m,source(ANALYSIS,'§raw packages: percentile endpoints; inference from 0.025/0.975 convention',95,'confidence convention; package endpoints transcribed, not re-bootstrapped'))

# Extra narrative C-A / C-D statistics introduced by the argument revision.
for pair in ('C-A','C-D'):
    for m in re.finditer(r"C\$-\$"+pair[-1]+r" gap is \$([-+]?\d+\.\d+)\$\s*\$\[([-+]?\d+\.\d+), ([-+]?\d+\.\d+)\]\$",text[results_start:results_end]):
        for i in range(3):
            start=results_start+m.start(i+1);end=results_start+m.end(i+1)
            put(tokens(start,end)[0],js(CE,'pairs',pair+' s0','tier2_bcplus',i))
# The original-paper baseline sentence is evidence outside our run packages.
# Only the original-paper QA means appear in the analysis §2 source passage.
if 'bounded-ratio GSPO baseline scores' in text[abstract_start:abstract_end]:
    a=text.index('bounded-ratio GSPO baseline scores',abstract_start)
    b=text.index('The boundedness finding',a)
    for m in tokens(a,b):
        assigned.pop(m.start(),None)
    corroborate(a,b,[ev('The paper reports 5-seed MHQA means','Our seed sds')]+header)
# No source in the three requested files gives TreeOPSD reward range 0.50–0.69.
# Same-valued QA/BC gaps elsewhere cannot validate training reward telemetry.
for m in re.finditer(r'TreeOPSD \(\$0\.50\$.*?\$0\.69\$',text[results_start:results_end]):
    for n in tokens(results_start+m.start(),results_start+m.end()):
        assigned.pop(n.start(),None)

audit_end=text.index('\\paragraph{Reproducibility')
corroborate(results_end,audit_end,header+ce_spans+cb_spans+mech+budget+selection+verdict)
for m in re.finditer(r'temperature \$0\.6\$, top-\$p\$ \$0\.95\$',text[results_end:audit_end]):
    for n in tokens(results_end+m.start(),results_end+m.end()):assigned.pop(n.start(),None)
# Recompute the concluding pooled gaps as well.
for metric,pat in [('tier1_qa',r'\+1\.98.*?\[\+1\.03, \+2\.94\]'),('tier2_bcplus',r'\+0\.08.*?\[-1\.24, \+1\.41\]')]:
    for m in re.finditer(pat,text[results_end:audit_end],re.S):
        sequence(m.group(),pooled(CE,metric),results_end+m.start())
# Bind all repeated pooled comparisons, including new mechanism paragraphs.
for a,b in [(abstract_start,abstract_end),(results_start,audit_end)]:
    for path,metric,pat in [
        (CE,'tier1_qa',r'\+1\.98\$\s*\$\[\+1\.03, \+2\.94\]'),
        (CE,'tier2_bcplus',r'\+0\.08\$\s*\$\[-1\.24, \+1\.41\]'),
    ]:
        for m in re.finditer(pat,text[a:b]):sequence(m.group(),pooled(path,metric),a+m.start())

for region,a,b in [('Abstract',abstract_start,abstract_end),('Results through conclusion (including Tables 1–2)',results_start,audit_end)]:
    for m in tokens(a,b):
        line=raw.count('\n',0,m.start())+1
        ref=assigned.get(m.start())
        claimed=number(m.group());decimals=len(m.group().split('.')[1]) if '.' in m.group() else 0
        if ref:
            value=ref['recomputed_value'];match=round(float(value),decimals)==round(claimed,decimals)
            status='MATCH' if match else 'MISMATCH'
        else:
            ref=source('', 'UNSUPPORTED: no mapped evidence in the three specified sources', '', 'PI evidence required')
            status='MISMATCH'
        rows.append(dict(region=region,paper_path=PAPER,paper_line=line,occurrence=len(rows)+1,
                         paper_value=m.group(),context=raw[max(a,m.start()-45):min(b,m.end()+55)].replace('\n',' '),
                         **ref,status=status))

# Fail closed on new/changed paper claims; never silently omit a numeric token.
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
paper_commit=subprocess.check_output(['git','log','-1','--format=%H','--',PAPER],cwd=ROOT,text=True).strip()
paper_dirty=bool(subprocess.check_output(['git','diff','HEAD','--',PAPER],cwd=ROOT))
hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [PAPER,CB,CE,ANALYSIS]}
fields=list(rows[0])
with (OUT/'number-check.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
bad=[r for r in rows if r['status']=='MISMATCH']
unsupported=[r for r in bad if not r['source_path']]
def cell(v):return str(v).replace('|','\\|').replace('\n',' ').strip()
with (OUT/'number-check.md').open('w') as f:
    f.write('# TreeOPSD number-check sheet\n\n')
    f.write(f'Generated {datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d PT")}. Latest paper commit `{paper_commit}`; working paper differs from that commit: `{paper_dirty}`. Input SHA-256 values below identify the exact audited bytes.\n\n')
    f.write(f'{len(rows)} occurrences: {len(rows)-len(bad)} MATCH, **{len(bad)} MISMATCH** ({len(unsupported)} lack mapped evidence).\n\n')
    f.write('Scope: every digit-bearing numeric token and the quantity words zero/one/two/three/four/five/sevenfold in the TeX abstract and Results, Mechanism, Discussion and Conclusion sections, including both complete tables, captions, seed labels, model identifiers and protocol labels. Repeated claims get separate rows. Table numbers are implicit in TeX and table labels identify them. TeX thousands separators are parsed as a single number. No paper text is edited.\n\n')
    f.write('JSON metrics override analysis prose. Run QA means, arm SDs, seed t intervals (two-sided 95% Student t critical, df=2), and two-seed gaps are recomputed. Pooled bootstrap CIs and other source-only values are transcribed, not independently re-estimated. MATCH on analysis-only rows means corroboration by that narrative; it does not certify raw logs. MISMATCH with no source is an evidence gap, not proof the claim is false. Seed t endpoints use the precise two-sided 95% critical value (4.302652729911275); the analysis rounds that constant to 4.30.\n\n')
    f.write('Drift audit: median-to-median drift increases 4.79/1.05 ≈ 4.56-fold; last-quarter drift increases 7.74/1.56 ≈ 4.96-fold. 7.74/1.05 ≈ 7.37 mixes time windows. If sevenfold wording appears in the audited text, it is flagged as MISMATCH. Structural identifier rows are included for coverage, not experimental findings.\n\n')
    f.write('## Input SHA-256\n\n'+''.join(f'- `{p}`: `{h}`\n' for p,h in hashes.items()))
    f.write('\n## Mismatches and evidence gaps\n\n')
    for r in bad:f.write(f'- Line {r["paper_line"]}, `{r["paper_value"]}`: {cell(r["source_key"])}; source/recomputed `{r["recomputed_value"]}`. Context: {cell(r["context"])}\n')
    f.write('\n## Full side-by-side inventory\n\n| # | Paper line | Paper value | Context | Source path and key | Source value | Recomputed value | Method | Status |\n|---|---|---|---|---|---|---|---|---|\n')
    for r in rows:f.write('| '+' | '.join(cell(x) for x in [r['occurrence'],r['paper_line'],r['paper_value'],r['context'],r['source_path']+' '+r['source_key'],r['source_value'],r['recomputed_value'],r['method'],r['status']])+' |\n')
print(json.dumps(dict(occurrences=len(rows),mismatches=len(bad),unsupported=len(unsupported),head=head)))
