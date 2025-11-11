#!/usr/bin/env python3
"""Script to run the full sampling (long runs).

Usage: adjust parameters in the __main__ block or call this script from the
command line after editing it. It will create ./samples and ./weights and
save pickles there.
"""
from pathlib import Path
import argparse
import logging
from RareEventSimulator import RareEventSimulator


def main(distance: int, p_target: float, p_known: float, num_seed: int, shots: int, burn_in: int, use_multiprocessing: bool):
    # Create simulator
    import stim

    circuit_p = lambda p: stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=distance,
        distance=distance,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )

    sim = RareEventSimulator(distance, circuit_p)

    # Ensure output directories
    Path("./samples").mkdir(parents=True, exist_ok=True)
    Path("./weights").mkdir(parents=True, exist_ok=True)

    sim.save_samples(p_target, p_known, num_seed, shots, burn_in, use_multiprocessing=use_multiprocessing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance', type=int, default=5)
    parser.add_argument('--p_target', type=float, default=1e-3)
    parser.add_argument('--p_known', type=float, default=4e-3)
    parser.add_argument('--num_seed', type=int, default=10)
    parser.add_argument('--shots', type=int, default=1000000)
    parser.add_argument('--burn_in', type=int, default=50000)
    parser.add_argument('--use_multiprocessing', action='store_true')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args.distance, args.p_target, args.p_known, args.num_seed, args.shots, args.burn_in, args.use_multiprocessing)
