"""
Test script for configuration system
"""

from src.config import Config

def main():
    print("Loading configuration...")
    config = Config()

    print("\n=== Paths Configuration ===")
    print(f"Project Root: {config.paths.project_root}")
    print(f"Audio Files: {config.paths.audio_files}")
    print(f"Transcripts: {config.paths.transcripts}")
    print(f"Output: {config.paths.output}")
    print(f"Training Root: {config.paths.training['root']}")
    print(f"FFmpeg Path: {config.get_ffmpeg_path()}")

    print("\n=== Models Configuration ===")
    print(f"WhisperX Enabled: {config.models.whisperx.enabled}")
    print(f"WhisperX Model: {config.models.whisperx.model}")
    print(f"WhisperX Device: {config.models.whisperx.device}")
    print(f"WhisperX Batch Size: {config.models.whisperx.batch_size}")
    print(f"Whisper Model: {config.models.whisper.model}")
    print(f"Pyannote Model: {config.models.pyannote.model}")

    print("\n=== Processing Configuration ===")
    print(f"Sample Rate: {config.processing.audio.sample_rate}")
    print(f"Output Format: {config.processing.output.format}")
    print(f"Calculate WER: {config.processing.metrics.calculate_wer}")
    print(f"Batch Enabled: {config.processing.batch.enabled}")

    print("\n=== Experiment Configuration ===")
    print(f"Experiment Name: {config.experiment.experiment_name}")
    print(f"Audio File: {config.experiment.audio_file}")
    print(f"Ground Truth: {config.experiment.transcript_truth}")

    print("\n=== Device Detection ===")
    device = config.get_device()
    print(f"Selected Device: {device}")

    print("\n[OK] Configuration system loaded successfully!")

if __name__ == "__main__":
    main()
