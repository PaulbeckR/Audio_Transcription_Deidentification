"""
Configuration Management Module

Loads and validates configuration from YAML files in the config/ directory.
Provides type-safe access to paths, model settings, and processing parameters.

Usage:
    from src.config import Config

    config = Config()
    audio_path = config.paths.audio_files
    model = config.models.whisperx.model
    batch_size = config.models.whisperx.batch_size
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Experiment configuration for test runs"""
    audio_file: str
    transcript_truth: Optional[str]
    experiment_name: str


@dataclass
class PathsConfig:
    """Path configuration"""
    project_root: str
    audio_files: str
    transcripts: str
    output: str
    training: Dict[str, str]
    temp: Dict[str, str]
    output_dirs: Dict[str, str]
    external: Dict[str, Optional[str]]

    def resolve(self, path: str) -> Path:
        """Resolve a relative path to absolute"""
        if os.path.isabs(path):
            return Path(path)
        return Path(self.project_root) / path


@dataclass
class WhisperXConfig:
    """WhisperX model configuration"""
    enabled: bool
    model: str
    device: str
    compute_type: str
    batch_size: int
    language: str
    alignment: Dict[str, Any]
    diarization: Dict[str, Any]
    model_name: str


@dataclass
class WhisperConfig:
    """Whisper model configuration"""
    model: str
    device: str
    language: str
    task: str
    word_timestamps: bool
    in_memory: bool


@dataclass
class PyannoteConfig:
    """Pyannote diarization configuration"""
    model: str
    device: str


@dataclass
class ModelsConfig:
    """Model configuration"""
    whisperx: WhisperXConfig
    whisper: WhisperConfig
    pyannote: PyannoteConfig
    auto_device: Dict[str, bool]
    memory: Dict[str, int]


@dataclass
class AudioConfig:
    """Audio preprocessing configuration"""
    sample_rate: int
    trim_silence: bool
    silence_threshold: int
    padding: float
    target_format: str
    channels: int


@dataclass
class AlignmentConfig:
    """Alignment configuration"""
    return_char_alignments: bool
    interpolate_method: str
    min_word_confidence: float
    min_segment_duration: float


@dataclass
class DiarizationConfig:
    """Diarization configuration"""
    enabled: bool
    min_speakers: Optional[int]
    max_speakers: Optional[int]
    min_segment_duration: float
    max_segment_duration: float


@dataclass
class OutputConfig:
    """Output format configuration"""
    format: str
    json: Dict[str, Any]
    csv: Dict[str, Any]
    naming: Dict[str, bool]


@dataclass
class MetricsConfig:
    """Metrics and evaluation configuration"""
    calculate_wer: bool
    calculate_der: bool
    ground_truth_format: str
    track_timing: bool
    track_memory: bool


@dataclass
class BatchConfig:
    """Batch processing configuration"""
    enabled: bool
    max_workers: int
    skip_existing: bool
    continue_on_error: bool


@dataclass
class TrainingExportConfig:
    """Training data export configuration"""
    enabled: bool
    format: str
    train_split: float
    val_split: float
    test_split: float
    min_audio_duration: float
    max_audio_duration: float
    min_wer: Optional[float]


@dataclass
class ProcessingConfig:
    """Processing configuration"""
    audio: AudioConfig
    alignment: AlignmentConfig
    diarization: DiarizationConfig
    output: OutputConfig
    metrics: MetricsConfig
    batch: BatchConfig
    training_export: TrainingExportConfig


@dataclass
class LoRAConfig:
    """LoRA configuration for parameter-efficient fine-tuning"""
    rank: int
    alpha: int
    dropout: float
    target_modules: list
    bias: str
    task_type: str
    presets: Dict[str, Dict[str, Any]]


@dataclass
class TrainingStrategyConfig:
    """Training strategy configuration"""
    method: str  # "lora", "full", "freeze_encoder"


@dataclass
class TrainingHyperparameters:
    """Training hyperparameters"""
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_epochs: int
    max_steps: int
    warmup_steps: int
    max_grad_norm: float
    early_stopping: Dict[str, Any]


@dataclass
class TrainingDatasetConfig:
    """Training dataset configuration"""
    input_dir: str
    output_dir: str
    splits: Dict[str, float]
    filters: Dict[str, Any]
    preprocessing: Dict[str, bool]


@dataclass
class TrainingConfig:
    """Training configuration"""
    strategy: TrainingStrategyConfig
    dataset: TrainingDatasetConfig
    hyperparameters: TrainingHyperparameters
    optimizer: Dict[str, Any]
    lr_scheduler: Dict[str, Any]
    mixed_precision: Dict[str, bool]
    freezing: Dict[str, bool]
    lora: LoRAConfig
    checkpoints: Dict[str, Any]
    evaluation: Dict[str, Any]
    logging: Dict[str, Any]
    memory: Dict[str, Any]
    quantization: Dict[str, bool]
    data_loader: Dict[str, Any]
    seed: int
    deterministic: bool
    distributed: Dict[str, Any]


@dataclass
class ExperimentDefinition:
    """Single experiment definition"""
    name: str
    description: str
    enabled: bool
    overrides: Dict[str, Any]


@dataclass
class ExperimentsConfig:
    """Experiments configuration for systematic testing"""
    active_experiment: str
    experiments: Dict[str, ExperimentDefinition]
    sweep: Dict[str, Any]
    datasets: Dict[str, Dict[str, Any]]
    active_dataset: str
    comparisons: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]


class Config:
    """
    Main configuration class that loads all YAML files.

    Attributes:
        paths: Path configuration
        models: Model configuration
        processing: Processing configuration
        experiment: Experiment/test run configuration
        training: Training configuration (hyperparameters, LoRA, etc.)
        experiments: Experiments configuration (experiment definitions, sweeps)

    Example:
        config = Config()

        # Access paths
        audio_dir = config.paths.audio_files

        # Access model settings
        model_name = config.models.whisperx.model
        batch_size = config.models.whisperx.batch_size

        # Access processing settings
        sample_rate = config.processing.audio.sample_rate

        # Access experiment settings
        test_name = config.experiment.experiment_name
        audio_file = config.experiment.audio_file

        # Access training settings
        learning_rate = config.training.hyperparameters.learning_rate
        lora_rank = config.training.lora.rank

        # Load specific experiment
        exp_config = config.get_experiment("baseline_lora")
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration from YAML files.

        Args:
            config_dir: Path to config directory (default: ./config)
        """
        if config_dir is None:
            # Default to config/ directory in project root
            project_root = Path(__file__).parent.parent
            config_dir = project_root / "config"
        else:
            config_dir = Path(config_dir)

        self.config_dir = config_dir

        # Load YAML files
        self._paths_data = self._load_yaml("paths.yaml")
        self._models_data = self._load_yaml("models.yaml")
        self._processing_data = self._load_yaml("processing.yaml")
        self._training_data = self._load_yaml("training.yaml")
        self._experiments_data = self._load_yaml("experiments.yaml")

        # Parse into dataclasses
        self.paths = self._parse_paths()
        self.models = self._parse_models()
        self.processing = self._parse_processing()
        self.training = self._parse_training()
        self.experiments = self._parse_experiments()
        # Backward compatibility: create experiment config from experiments data if needed
        self.experiment = self._parse_experiment()

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML file from config directory"""
        filepath = self.config_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        return data if data is not None else {}

    def _parse_paths(self) -> PathsConfig:
        """Parse paths configuration"""
        return PathsConfig(
            project_root=self._paths_data.get('project_root', '.'),
            audio_files=self._paths_data.get('audio_files', ''),
            transcripts=self._paths_data.get('transcripts', ''),
            output=self._paths_data.get('output', ''),
            training=self._paths_data.get('training', {}),
            temp=self._paths_data.get('temp', {}),
            output_dirs=self._paths_data.get('output_dirs', {}),
            external=self._paths_data.get('external', {}),
        )

    def _parse_experiment(self) -> ExperimentConfig:
        """
        Parse experiment configuration (backward compatibility).

        Note: This uses experiments.yaml now. The old experiment.yaml is no longer needed.
        """
        # Try to get from experiments data, otherwise use defaults
        exp_data = self._experiments_data.get('current_experiment', {})
        return ExperimentConfig(
            audio_file=exp_data.get('audio_file', ''),
            transcript_truth=exp_data.get('transcript_truth'),
            experiment_name=exp_data.get('experiment_name', 'default'),
        )

    def _parse_models(self) -> ModelsConfig:
        """Parse models configuration"""
        whisperx_data = self._models_data.get('whisperx', {})
        whisper_data = self._models_data.get('whisper', {})
        pyannote_data = self._models_data.get('pyannote', {})

        whisperx = WhisperXConfig(
            enabled=whisperx_data.get('enabled', True),
            model=whisperx_data.get('model', 'large-v2'),
            device=whisperx_data.get('device', 'cuda'),
            compute_type=whisperx_data.get('compute_type', 'float16'),
            batch_size=whisperx_data.get('batch_size', 4),
            language=whisperx_data.get('language', 'en'),
            alignment=whisperx_data.get('alignment', {}),
            diarization=whisperx_data.get('diarization', {}),
            model_name=whisper_data.get('model_name', "openai/whisper-large-v2"),

        )

        whisper = WhisperConfig(
            model=whisper_data.get('model', 'large-v2'),
            device=whisper_data.get('device', 'cuda'),
            language=whisper_data.get('language', 'en'),
            task=whisper_data.get('task', 'transcribe'),
            word_timestamps=whisper_data.get('word_timestamps', True),
            in_memory=whisper_data.get('in_memory', True),
        )

        pyannote = PyannoteConfig(
            model=pyannote_data.get('model', 'pyannote/speaker-diarization-3.1'),
            device=pyannote_data.get('device', 'cuda'),
        )

        return ModelsConfig(
            whisperx=whisperx,
            whisper=whisper,
            pyannote=pyannote,
            auto_device=self._models_data.get('auto_device', {}),
            memory=self._models_data.get('memory', {}),
        )

    def _parse_processing(self) -> ProcessingConfig:
        """Parse processing configuration"""
        audio_data = self._processing_data.get('audio', {})
        alignment_data = self._processing_data.get('alignment', {})
        diarization_data = self._processing_data.get('diarization', {})
        output_data = self._processing_data.get('output', {})
        metrics_data = self._processing_data.get('metrics', {})
        batch_data = self._processing_data.get('batch', {})
        training_data = self._processing_data.get('training_export', {})

        audio = AudioConfig(
            sample_rate=audio_data.get('sample_rate', 16000),
            trim_silence=audio_data.get('trim_silence', True),
            silence_threshold=audio_data.get('silence_threshold', -40),
            padding=audio_data.get('padding', 0.5),
            target_format=audio_data.get('target_format', 'wav'),
            channels=audio_data.get('channels', 1),
        )

        alignment = AlignmentConfig(
            return_char_alignments=alignment_data.get('return_char_alignments', False),
            interpolate_method=alignment_data.get('interpolate_method', 'nearest'),
            min_word_confidence=alignment_data.get('min_word_confidence', 0.0),
            min_segment_duration=alignment_data.get('min_segment_duration', 0.1),
        )

        diarization = DiarizationConfig(
            enabled=diarization_data.get('enabled', True),
            min_speakers=diarization_data.get('min_speakers'),
            max_speakers=diarization_data.get('max_speakers'),
            min_segment_duration=diarization_data.get('min_segment_duration', 0.5),
            max_segment_duration=diarization_data.get('max_segment_duration', 30.0),
        )

        output = OutputConfig(
            format=output_data.get('format', 'json'),
            json=output_data.get('json', {}),
            csv=output_data.get('csv', {}),
            naming=output_data.get('naming', {}),
        )

        metrics = MetricsConfig(
            calculate_wer=metrics_data.get('calculate_wer', True),
            calculate_der=metrics_data.get('calculate_der', False),
            ground_truth_format=metrics_data.get('ground_truth_format', 'xlsx'),
            track_timing=metrics_data.get('track_timing', True),
            track_memory=metrics_data.get('track_memory', True),
        )

        batch = BatchConfig(
            enabled=batch_data.get('enabled', False),
            max_workers=batch_data.get('max_workers', 1),
            skip_existing=batch_data.get('skip_existing', True),
            continue_on_error=batch_data.get('continue_on_error', True),
        )

        training_export = TrainingExportConfig(
            enabled=training_data.get('enabled', False),
            format=training_data.get('format', 'huggingface'),
            train_split=training_data.get('train_split', 0.8),
            val_split=training_data.get('val_split', 0.1),
            test_split=training_data.get('test_split', 0.1),
            min_audio_duration=training_data.get('min_audio_duration', 1.0),
            max_audio_duration=training_data.get('max_audio_duration', 30.0),
            min_wer=training_data.get('min_wer'),
        )

        return ProcessingConfig(
            audio=audio,
            alignment=alignment,
            diarization=diarization,
            output=output,
            metrics=metrics,
            batch=batch,
            training_export=training_export,
        )

    def get_ffmpeg_path(self) -> Optional[str]:
        """Get FFmpeg path from configuration"""
        return self.paths.external.get('ffmpeg')

    def get_hf_token(self) -> Optional[str]:
        """Get Hugging Face token from environment variable"""
        return os.environ.get('HF_TOKEN')

    def get_device(self) -> str:
        """
        Get device to use (cuda or cpu).
        Checks if CUDA is available if device is set to cuda.
        """
        device = self.models.whisperx.device

        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    if self.models.auto_device.get('fallback_to_cpu', True):
                        print("CUDA not available, falling back to CPU")
                        return "cpu"
                    else:
                        raise RuntimeError("CUDA not available and fallback disabled")
            except ImportError:
                print("PyTorch not installed, cannot check CUDA availability")
                return "cpu"

        return device

    def _parse_training(self) -> TrainingConfig:
        """Parse training configuration"""
        strategy_data = self._training_data.get('strategy', {})
        dataset_data = self._training_data.get('dataset', {})
        hyperparams_data = self._training_data.get('hyperparameters', {})
        lora_data = self._training_data.get('lora', {})

        strategy = TrainingStrategyConfig(
            method=strategy_data.get('method', 'lora')
        )

        dataset = TrainingDatasetConfig(
            input_dir=dataset_data.get('input_dir', 'datasets/whisper_finetuning'),
            output_dir=dataset_data.get('output_dir', 'models/whisper-finetuned'),
            splits=dataset_data.get('splits', {}),
            filters=dataset_data.get('filters', {}),
            preprocessing=dataset_data.get('preprocessing', {})
        )

        hyperparameters = TrainingHyperparameters(
            batch_size=hyperparams_data.get('batch_size', 2),
            gradient_accumulation_steps=hyperparams_data.get('gradient_accumulation_steps', 8),
            learning_rate=hyperparams_data.get('learning_rate', 1e-5),
            num_epochs=hyperparams_data.get('num_epochs', 5),
            max_steps=hyperparams_data.get('max_steps', -1),
            warmup_steps=hyperparams_data.get('warmup_steps', 100),
            max_grad_norm=hyperparams_data.get('max_grad_norm', 1.0),
            early_stopping=hyperparams_data.get('early_stopping', {})
        )

        lora = LoRAConfig(
            rank=lora_data.get('rank', 32),
            alpha=lora_data.get('alpha', 64),
            dropout=lora_data.get('dropout', 0.05),
            target_modules=lora_data.get('target_modules', ['q_proj', 'v_proj']),
            bias=lora_data.get('bias', 'none'),
            task_type=lora_data.get('task_type', 'CAUSAL_LM'),
            presets=lora_data.get('presets', {})
        )

        return TrainingConfig(
            strategy=strategy,
            dataset=dataset,
            hyperparameters=hyperparameters,
            optimizer=self._training_data.get('optimizer', {}),
            lr_scheduler=self._training_data.get('lr_scheduler', {}),
            mixed_precision=self._training_data.get('mixed_precision', {}),
            freezing=self._training_data.get('freezing', {}),
            lora=lora,
            checkpoints=self._training_data.get('checkpoints', {}),
            evaluation=self._training_data.get('evaluation', {}),
            logging=self._training_data.get('logging', {}),
            memory=self._training_data.get('memory', {}),
            quantization=self._training_data.get('quantization', {}),
            data_loader=self._training_data.get('data_loader', {}),
            seed=self._training_data.get('seed', 42),
            deterministic=self._training_data.get('deterministic', False),
            distributed=self._training_data.get('distributed', {})
        )

    def _parse_experiments(self) -> ExperimentsConfig:
        """Parse experiments configuration"""
        experiments_dict = {}
        raw_experiments = self._experiments_data.get('experiments', {})

        for exp_name, exp_data in raw_experiments.items():
            experiments_dict[exp_name] = ExperimentDefinition(
                name=exp_data.get('name', exp_name),
                description=exp_data.get('description', ''),
                enabled=exp_data.get('enabled', True),
                overrides=exp_data  # Store full experiment data for merging
            )

        return ExperimentsConfig(
            active_experiment=self._experiments_data.get('active_experiment', 'baseline_lora'),
            experiments=experiments_dict,
            sweep=self._experiments_data.get('sweep', {}),
            datasets=self._experiments_data.get('datasets', {}),
            active_dataset=self._experiments_data.get('active_dataset', 'gpt_test'),
            comparisons=self._experiments_data.get('comparisons', {}),
            metadata=self._experiments_data.get('metadata', {})
        )

    def get_experiment(self, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get experiment configuration with overrides applied.

        Args:
            experiment_name: Name of experiment to load (default: active_experiment)

        Returns:
            Merged configuration dict with experiment overrides applied

        Example:
            config = Config()
            exp = config.get_experiment("baseline_lora")
            learning_rate = exp['hyperparameters']['learning_rate']
        """
        if experiment_name is None:
            experiment_name = self.experiments.active_experiment

        if experiment_name not in self.experiments.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found in experiments.yaml")

        experiment = self.experiments.experiments[experiment_name]

        if not experiment.enabled:
            print(f"Warning: Experiment '{experiment_name}' is disabled")

        # Return the raw overrides dict for now
        # In usage, caller will merge with base training config
        return experiment.overrides

    def get_dataset_config(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get dataset configuration.

        Args:
            dataset_name: Name of dataset (default: active_dataset)

        Returns:
            Dataset configuration dict
        """
        if dataset_name is None:
            dataset_name = self.experiments.active_dataset

        if dataset_name not in self.experiments.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in experiments.yaml")

        return self.experiments.datasets[dataset_name]

    def __repr__(self) -> str:
        return f"Config(config_dir='{self.config_dir}')"


# Convenience function for loading default config
def load_config(config_dir: Optional[str] = None) -> Config:
    """
    Load configuration from YAML files.

    Args:
        config_dir: Path to config directory (default: ./config)

    Returns:
        Config object with all settings loaded

    Example:
        config = load_config()
        print(config.models.whisperx.model)
    """
    return Config(config_dir=config_dir)
