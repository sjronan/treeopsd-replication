"""Print the non-base requirements of packages installed with --no-deps (build step 2b).

Base-pinned names (torch, transformers, nvidia-*, ...) are skipped so no resolver touches them;
everything else a package declares (for the current platform, no extras) is printed for uv.
"""
import re, sys
from importlib.metadata import distribution
from packaging.requirements import Requirement

base = {re.split(r'==', l.strip())[0].lower().replace('_', '-')
        for l in open('/etc/treeopsd-base-constraints.txt') if '==' in l}
out = set()
for name in sys.argv[1:]:
    for r in distribution(name).requires or []:
        req = Requirement(r)
        if req.marker and not req.marker.evaluate({'extra': ''}):
            continue
        key = req.name.lower().replace('_', '-')
        if key in base or key.startswith('nvidia-'):
            continue
        out.add(str(req).split(';')[0].strip())
print('\n'.join(sorted(out)))
