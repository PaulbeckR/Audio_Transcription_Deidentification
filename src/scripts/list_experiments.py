"""
List all available experiments and their configurations.

Usage:
    python src/scripts/list_experiments.py
    python src/scripts/list_experiments.py --verbose
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Config


def print_experiment_details(exp_name: str, exp_def, is_active: bool = False, verbose: bool = False):
    """Print detailed experiment information."""
    status = "[ACTIVE]" if is_active else "[      ]"
    enabled = "[ENABLED]  " if exp_def.enabled else "[DISABLED] "

    print(f"\n{status} {enabled} {exp_name}")
    print(f"  Name: {exp_def.name}")
    print(f"  Description: {exp_def.description}")

    if verbose or is_active:
        overrides = exp_def.overrides

        # Strategy
        if 'strategy' in overrides:
            method = overrides['strategy'].get('method', 'N/A')
            print(f"\n  Strategy:")
            print(f"    Method: {method}")

        # Hyperparameters
        if 'hyperparameters' in overrides:
            hp = overrides['hyperparameters']
            print(f"\n  Hyperparameters:")
            for key, value in hp.items():
                print(f"    {key}: {value}")

        # LoRA config
        if 'lora' in overrides:
            lora = overrides['lora']
            print(f"\n  LoRA Configuration:")
            for key, value in lora.items():
                if key != 'presets':  # Skip presets dict
                    print(f"    {key}: {value}")

        # Dataset
        if 'dataset' in overrides:
            ds = overrides['dataset']
            print(f"\n  Dataset:")
            for key, value in ds.items():
                print(f"    {key}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="List available experiments")
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed configuration')
    parser.add_argument('--active-only', action='store_true', help='Only show active experiment')

    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    try:
        config = Config()
        print("OK: Configuration loaded\n")
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        sys.exit(1)

    active = config.experiments.active_experiment

    print("=" * 80)
    print("AVAILABLE EXPERIMENTS")
    print("=" * 80)

    if args.active_only:
        # Show only active experiment
        if active in config.experiments.experiments:
            exp_def = config.experiments.experiments[active]
            print_experiment_details(active, exp_def, is_active=True, verbose=True)
    else:
        # Show all experiments
        for exp_name, exp_def in config.experiments.experiments.items():
            is_active = (exp_name == active)
            print_experiment_details(exp_name, exp_def, is_active=is_active, verbose=args.verbose)

    # Summary
    print("\n" + "=" * 80)
    print(f"Active experiment: {active}")
    print(f"Total experiments: {len(config.experiments.experiments)}")
    enabled_count = sum(1 for exp in config.experiments.experiments.values() if exp.enabled)
    print(f"Enabled: {enabled_count}")
    print("=" * 80)

    print("\nUsage:")
    print("  python src/scripts/run_training.py                    # Run active experiment")
    print("  python src/scripts/run_training.py --experiment NAME  # Run specific experiment")
    print("  python src/scripts/run_training.py --list             # List with run_training.py")


if __name__ == "__main__":
    main()
