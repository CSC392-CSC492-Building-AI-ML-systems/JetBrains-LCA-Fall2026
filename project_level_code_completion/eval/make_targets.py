"""Write the targeted lines (loose X(L), research log section 8.13) of a split to a JSON file, for eval.analyze_arms.

The result maps str(datapoint position within the split) -> {line_class: [line numbers]}, which is exactly the
`datapoint_idx` that `evaluate_generation` stores for that split. It uses only the repository snapshot and the file
prefix, never any model output.

Example:
    python -m eval.make_targets --dataset medium --split heldout --dev-size 28 --dev-extra 28 --out medium_heldout.json
"""
import argparse
import json
import os

from eval.line_generators import dev_indices, heldout_indices
from eval.proxy_metrics import targeted_lines


def targets_for(datapoints: list, split: str, dev_size: int, dev_extra: int) -> dict:
    if split == 'dev':
        indices = dev_indices(len(datapoints), dev_size, dev_extra)
    elif split == 'heldout':
        indices = heldout_indices(len(datapoints), dev_size, dev_extra)
    else:
        raise ValueError(f"split must be dev or heldout, got {split!r}")
    return {str(position): targeted_lines(datapoints[i]) for position, i in enumerate(indices)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=['small', 'medium', 'large', 'huge'])
    parser.add_argument('--split', required=True, choices=['dev', 'heldout'])
    parser.add_argument('--dev-size', type=int, default=20)
    parser.add_argument('--dev-extra', type=int, default=28)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir

    from eval.preprocessors import load_datapoints

    with initialize_config_dir(config_dir=os.path.join(os.path.dirname(__file__), 'config'), version_base=None):
        cfg = compose(config_name='compare_arms', overrides=[f'dataset={args.dataset}'])
    targets = targets_for(load_datapoints(cfg.dataset), args.split, args.dev_size, args.dev_extra)
    with open(os.path.expanduser(args.out), 'w') as f:
        json.dump(targets, f)
    n_lines = sum(len(v) for t in targets.values() for v in t.values())
    print(f"{args.dataset}/{args.split}: {len(targets)} datapoints, {n_lines} targeted lines -> {args.out}")


if __name__ == '__main__':
    main()
