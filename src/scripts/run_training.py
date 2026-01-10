"""
Run fine-tuning experiments using configuration files.

This script provides the main entry point for running Whisper fine-tuning
experiments. All parameters are controlled via config files.

Usage:
    # Run active experiment
    python src/scripts/run_training.py

    # Run specific experiment
    python src/scripts/run_training.py --experiment baseline_lora

    # Run with LoRA preset
    python src/scripts/run_training.py --experiment baseline_lora --preset medium

    # List available experiments
    python src/scripts/run_training.py --list

Examples:
    # Quick test (1 epoch, minimal LoRA)
    python src/scripts/run_training.py --experiment quick_test

    # Baseline LoRA (matches Dec 29 success)
    python src/scripts/run_training.py --experiment baseline_lora

    # Full fine-tuning (no LoRA)
    python src/scripts/run_training.py --experiment full_finetune

    # High-capacity LoRA
    python src/scripts/run_training.py --experiment lora_large
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# CRITICAL: Apply torch.load patch BEFORE importing any torch-dependent modules
# This fixes PyTorch 2.6+ compatibility with WhisperX/Pyannote models
from src.core.torch_utils import patch_torch_load
patch_torch_load()

from src.config import Config


def list_experiments(config: Config):
    """
    List all available experiments.

    Args:
        config: Config object
    """
    print("\n" + "=" * 80)
    print("AVAILABLE EXPERIMENTS")
    print("=" * 80)

    active = config.experiments.active_experiment

    for exp_name, exp_def in config.experiments.experiments.items():
        status = "[ACTIVE]" if exp_name == active else "[      ]"
        enabled = "[x]" if exp_def.enabled else "[ ]"
        print(f"\n{status} {enabled} {exp_name}")
        print(f"  Name: {exp_def.name}")
        print(f"  Description: {exp_def.description}")

        # Show key settings
        overrides = exp_def.overrides
        if 'strategy' in overrides:
            print(f"  Strategy: {overrides['strategy'].get('method', 'N/A')}")
        if 'hyperparameters' in overrides:
            hp = overrides['hyperparameters']
            if 'learning_rate' in hp:
                print(f"  Learning rate: {hp['learning_rate']}")
            if 'num_epochs' in hp:
                print(f"  Epochs: {hp['num_epochs']}")
        if 'lora' in overrides:
            lora = overrides['lora']
            if 'rank' in lora:
                print(f"  LoRA rank: {lora['rank']}")

    print("\n" + "=" * 80)
    print(f"\nActive experiment: {active}")
    print("\nUsage:")
    print("  python src/scripts/run_training.py                    # Run active")
    print("  python src/scripts/run_training.py --experiment NAME  # Run specific")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Whisper fine-tuning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--experiment',
        type=str,
        help='Name of experiment to run (uses active if not specified)'
    )

    parser.add_argument(
        '--preset',
        type=str,
        choices=['minimal', 'small', 'medium', 'large'],
        help='LoRA preset to use (overrides config)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available experiments and exit'
    )

    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    try:
        config = Config()
        print("  OK: Configuration loaded")
    except Exception as e:
        print(f"  ERROR: Failed to load configuration: {e}")
        sys.exit(1)

    # List experiments if requested
    if args.list:
        list_experiments(config)
        return

    # Determine experiment to run
    experiment_name = args.experiment
    if experiment_name is None:
        experiment_name = config.experiments.active_experiment
        print(f"\nUsing active experiment: {experiment_name}")
    else:
        print(f"\nRunning experiment: {experiment_name}")

    # Validate experiment exists
    if experiment_name not in config.experiments.experiments:
        print(f"  ERROR: Experiment '{experiment_name}' not found")
        print("\nAvailable experiments:")
        for name in config.experiments.experiments.keys():
            print(f"  - {name}")
        sys.exit(1)

    # Check if experiment is enabled
    exp_def = config.experiments.experiments[experiment_name]
    if not exp_def.enabled:
        print(f"  WARNING: Experiment '{experiment_name}' is disabled in config")
        response = input("  Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("  Aborted")
            sys.exit(0)

    # Run fine-tuning
    try:
        # Lazy import: only import finetune when actually training
        # This allows --list to work without installing all dependencies
        try:
            from src.training.finetune import finetune_whisper
        except ImportError as e:
            print(f"\n  ERROR: Training dependencies not installed")
            print(f"  Install with: pip install -r requirements_finetuning.txt")
            print(f"  Original error: {e}")
            sys.exit(1)

        model, trainer = finetune_whisper(
            config,
            experiment_name=experiment_name,
            preset=args.preset
        )

        print("\n" + "=" * 80)
        print("SUCCESS")
        print("=" * 80)
        print(f"\nExperiment '{experiment_name}' completed successfully!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
