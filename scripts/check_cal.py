import csv, sys

fn = sys.argv[1] if len(sys.argv) > 1 else "sweep_period_cal3.csv"
rows = list(csv.DictReader(open(fn)))
cols = list(rows[0].keys())
param = cols[0]

print(f"{'Period':>6s} {'Ctrl':>14s} {'barrier':>8s} {'tail_b':>8s} {'effort':>8s} {'E_core':>8s} {'fm':>6s} {'uh':>3s}")
for r in rows:
    print(f"{float(r[param]):6.1f} {r['controller']:>14s} "
          f"{float(r['avg_barrier_aniso']):8.3f} {float(r['tail_barrier']):8.3f} "
          f"{float(r['avg_effort']):8.3f} {float(r['avg_E_core']):8.2f} "
          f"{float(r['fusion_margin']):6.2f} {r['underheat']:>3s}")
