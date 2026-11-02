"""Controlled benchmark and ablations. CPU default; no network access."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from .train import train_one

DEFAULT_VARIANTS = ['cnn', 'vit8', 'vit4', 'vit8_no_pos', 'vit8_mean']


def run(seeds=3, steps=150, train_count=1200, eval_count=128,
        variants=DEFAULT_VARIANTS, device='cpu', output='results/benchmark.json'):
    rows = []
    for name in variants:
        for seed in range(seeds):
            row = train_one(name, seed, steps, train_count, eval_count=eval_count, device=device)
            print(f"{name:12s} seed={seed} clean={row['clean_accuracy']:.3f} noise={row['noisy_accuracy']:.3f} shift={row['shifted_accuracy']:.3f} seconds={row['train_seconds']:.2f}", flush=True)
            rows.append(row)
    grouped = {}
    for name in variants:
        subset = [r for r in rows if r['name'] == name]
        grouped[name] = {}
        for metric in ('clean_accuracy','noisy_accuracy','shifted_accuracy','train_seconds'):
            vals = [r[metric] for r in subset]
            grouped[name][metric] = {'mean':mean(vals),'sample_std':stdev(vals) if len(vals)>1 else None}
    result={'description':'Synthetic two-marker spatial relation, independent PyTorch educational benchmark',
            'not_a_replication_of_original_paper':True,
            'configuration':{'seeds':seeds,'steps':steps,'train_count':train_count,'eval_count':eval_count,'variants':variants,'device':device},
            'runs': rows, 'summary':grouped}
    path=Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--train-count', type=int, default=1200)
    p.add_argument('--eval-count', type=int, default=128)
    p.add_argument('--variants', nargs='+', default=DEFAULT_VARIANTS)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', default='results/benchmark.json')
    args=p.parse_args()
    run(args.seeds, args.steps, args.train_count, args.eval_count, args.variants, args.device, args.output)

if __name__ == '__main__': main()
