import argparse

from .retouch import retouch
from .workflow import execute

p = argparse.ArgumentParser(
    description="Prepare dress color proofs or render shop-review candidates"
)
p.add_argument("action", choices=["prepare", "render", "retouch"])
p.add_argument("job")
p.add_argument("--out", required=True)
p.add_argument("--only", help="Render one observed color ID, e.g. rack-01-03")
args = p.parse_args()
if args.action == "retouch":
    m = retouch(args.job, args.out, args.only)
else:
    m = execute(args.job, args.out, args.action == "render", args.only)
print(f"{m['style_id']} | {m['state']} | {len(m['variants'])} color observations")
