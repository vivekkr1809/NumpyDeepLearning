"""Experiment tracking system."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np


class ExperimentTracker:
    """
    Track experiments, metrics, and hyperparameters.

    Saves experiment results to disk for later analysis.
    """

    def __init__(self, experiment_name: str, save_dir: str = './experiments'):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            save_dir: Directory to save experiment results
        """
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = datetime.now()
        self.metrics_history = {}
        self.hyperparameters = {}
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': self.start_time.isoformat(),
        }

    def log_hyperparameters(self, **hyperparameters):
        """
        Log hyperparameters.

        Args:
            **hyperparameters: Hyperparameter name-value pairs
        """
        self.hyperparameters.update(hyperparameters)
        self._save_hyperparameters()

    def log_metrics(self, step: int, **metrics):
        """
        Log metrics at a specific step.

        Args:
            step: Step/epoch number
            **metrics: Metric name-value pairs
        """
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []

            # Convert numpy types to Python types
            if isinstance(value, np.ndarray):
                value = value.item() if value.size == 1 else value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                value = value.item()

            self.metrics_history[name].append({
                'step': step,
                'value': value,
                'timestamp': datetime.now().isoformat(),
            })

        self._save_metrics()

    def log_artifact(self, artifact_name: str, artifact: Any):
        """
        Log an artifact (e.g., model weights, plots).

        Args:
            artifact_name: Name of the artifact
            artifact: Artifact to save
        """
        artifact_path = self.save_dir / 'artifacts' / artifact_name
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(artifact, dict):
            # Save as JSON
            with open(artifact_path.with_suffix('.json'), 'w') as f:
                json.dump(artifact, f, indent=2)
        elif isinstance(artifact, np.ndarray):
            # Save as numpy file
            np.save(artifact_path.with_suffix('.npy'), artifact)
        else:
            # Save as text
            with open(artifact_path.with_suffix('.txt'), 'w') as f:
                f.write(str(artifact))

    def log_model_checkpoint(self, model_state: Dict[str, Any], epoch: int, metrics: Dict[str, float] = None):
        """
        Log a model checkpoint.

        Args:
            model_state: Model state dictionary
            epoch: Epoch number
            metrics: Optional metrics at this checkpoint
        """
        checkpoint_dir = self.save_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.npz'

        # Save checkpoint
        np.savez(checkpoint_path, **model_state)

        # Save checkpoint metadata
        metadata = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {},
        }

        metadata_path = checkpoint_dir / f'checkpoint_epoch_{epoch}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved checkpoint to {checkpoint_path}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics.

        Returns:
            Dictionary with metric summaries
        """
        summary = {}
        for metric_name, history in self.metrics_history.items():
            values = [entry['value'] for entry in history]
            summary[metric_name] = {
                'current': values[-1] if values else None,
                'best': max(values) if values else None,
                'worst': min(values) if values else None,
                'mean': np.mean(values) if values else None,
                'std': np.std(values) if values else None,
            }
        return summary

    def finish(self, status: str = 'completed'):
        """
        Finish the experiment and save final metadata.

        Args:
            status: Final status of the experiment
        """
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['duration'] = str(datetime.now() - self.start_time)
        self.metadata['status'] = status
        self.metadata['metrics_summary'] = self.get_metrics_summary()

        self._save_metadata()
        print(f"Experiment '{self.experiment_name}' finished with status: {status}")
        print(f"Results saved to: {self.save_dir}")

    def _save_hyperparameters(self):
        """Save hyperparameters to file."""
        hparams_path = self.save_dir / 'hyperparameters.json'
        with open(hparams_path, 'w') as f:
            json.dump(self.hyperparameters, f, indent=2)

    def _save_metrics(self):
        """Save metrics history to file."""
        metrics_path = self.save_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def _save_metadata(self):
        """Save metadata to file."""
        metadata_path = self.save_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load(cls, experiment_name: str, save_dir: str = './experiments') -> 'ExperimentTracker':
        """
        Load an existing experiment.

        Args:
            experiment_name: Name of the experiment
            save_dir: Directory where experiments are saved

        Returns:
            ExperimentTracker instance
        """
        tracker = cls(experiment_name, save_dir)

        # Load hyperparameters
        hparams_path = tracker.save_dir / 'hyperparameters.json'
        if hparams_path.exists():
            with open(hparams_path, 'r') as f:
                tracker.hyperparameters = json.load(f)

        # Load metrics
        metrics_path = tracker.save_dir / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                tracker.metrics_history = json.load(f)

        # Load metadata
        metadata_path = tracker.save_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                tracker.metadata = json.load(f)

        return tracker


class Logger:
    """Simple logger for training progress."""

    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize logger.

        Args:
            log_file: Path to log file (prints to console if None)
        """
        self.log_file = log_file
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str):
        """
        Log a message.

        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"

        print(log_message)

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_message + '\n')

    def info(self, message: str):
        """Log info message."""
        self.log(f"INFO: {message}")

    def warning(self, message: str):
        """Log warning message."""
        self.log(f"WARNING: {message}")

    def error(self, message: str):
        """Log error message."""
        self.log(f"ERROR: {message}")
