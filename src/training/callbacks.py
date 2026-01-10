"""
Training callbacks for monitoring and logging during fine-tuning.
"""

import torch
from transformers import TrainerCallback


class MemoryMonitorCallback(TrainerCallback):
    """
    Monitor and log GPU memory usage during training.

    Logs allocated and reserved GPU memory every N steps.
    """

    def __init__(self, log_interval: int = 10):
        """
        Initialize memory monitor.

        Args:
            log_interval: Log memory every N steps
        """
        self.log_interval = log_interval

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        if torch.cuda.is_available() and state.global_step % self.log_interval == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            print(
                f"  [Step {state.global_step}] "
                f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
            )


class WERTrackingCallback(TrainerCallback):
    """
    Track WER improvements during training.

    Logs WER metric and reports improvements.
    """

    def __init__(self):
        """Initialize WER tracker."""
        self.best_wer = float('inf')
        self.wer_history = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics and 'eval_wer' in metrics:
            current_wer = metrics['eval_wer']
            self.wer_history.append(current_wer)

            if current_wer < self.best_wer:
                improvement = self.best_wer - current_wer
                self.best_wer = current_wer
                print(f"\n  New best WER: {current_wer:.4f} (improved by {improvement:.4f})")
            else:
                print(f"\n  Current WER: {current_wer:.4f} (best: {self.best_wer:.4f})")


class ExperimentTrackingCallback(TrainerCallback):
    """
    Track experiment metadata and configuration.

    Logs experiment name, configuration, and training progress.
    """

    def __init__(self, experiment_name: str, config: dict):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary
        """
        self.experiment_name = experiment_name
        self.config = config

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {self.experiment_name}")
        print(f"{'='*80}")

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETED: {self.experiment_name}")
        print(f"Total steps: {state.global_step}")
        print(f"{'='*80}")
