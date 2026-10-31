"""Run the predeclared, checkpointed W2 study."""
import argparse
import os
from pathlib import Path
import sys
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from nhanes_pipeline.w2 import run
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--outer', type=int, default=5)
    p.add_argument('--inner', type=int, default=3)
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--output', type=Path, help='Separate output directory for a distinct protocol')
    a = p.parse_args()
    run(ROOT, a.repeats, a.outer, a.inner, a.jobs, a.output)
