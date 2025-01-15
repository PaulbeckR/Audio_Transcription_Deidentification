
import ffmpeg
from pydub import AudioSegment
import pydub
import shutil
import numpy as np
import librosa

__all__ = ['convert_to_wav', 'millisec', 'extract_end_time', 'milliseconds_until_sound', 'move_file', 'analyze_chunks', 
           'samples_per_ms', 'adjust_to_frame_length', 'compute_dynamic_thresholds']

  
# def convert_to_wav(input_file, output_file):
#     try:
#         stream = ffmpeg.input(input_file)
#         stream = ffmpeg.output(stream, output_file)
#         ffmpeg.run(stream)
#        # print(f"Conversion successful: {output_file}")
#     except ffmpeg.Error as e:
#         print(f"Error occurred: {e}")


def convert_to_wav(input, output):
    try:
        audio = AudioSegment.from_file(input)
        audio.export(output, format='wav')
    except Exception as e:
        print("error during converstion: {e}")

    

def millisec(timeStr):
   # print("timestr", timeStr)
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )*1000)

def extract_end_time(rttm_line):
#print("extract end time line", rttm_line)
    tokens = rttm_line.strip().split()
   # print("tokens",tokens, tokens[3], tokens[4], tokens[5])
    start_time = float(tokens[3])
    duration = float(tokens[4])
    end_time = start_time + duration
    return start_time, end_time, duration

#Trim Start silence (enhance transcription method)
def milliseconds_until_sound(sound, silence_threshold_in_decibels = -20.0, chunk_size=10):
    trim_ms = 0
    assert chunk_size > 0
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold_in_decibels and trim_ms < len(sound):
        trim_ms += chunk_size
    return trim_ms


def move_file(source_file_path, destination_folder_path):

    try:
        shutil.move(source_file_path, destination_folder_path)

    except Exception as e:
        print(f"error moving file {e}")


#Finetuning-openais-whisper-creating-your-custom-dataset

def adjust_to_frame_length(chunk_length_ms, frame_length_ms):
    return ((chunk_length_ms + frame_length_ms - 1) // frame_length_ms ) * frame_length_ms

def samples_per_ms(sample_rate, ms):
    return int(sample_rate * ms / 1000)

def compute_dynamic_thresholds(audio, sr, frame_length_samples, segment_length_samples):
    """
    Computes thresholds for energy and Zero-Crossing Rate (ZCR).

    The thresholds are determined based on the mean and standard deviation
    of the energy and ZCR within individual segments of the audio signal.
    """

    # Initialization of lists for energy thresholds and ZCR thresholds
    energy_thresholds = []
    zcr_thresholds = []

    # Iterating over the audio signal in segments
    for i in range(0, len(audio), segment_length_samples):
        # Extract the current segment from the audio signal
        segment = audio[i:i + segment_length_samples]
        
        # Calculate the number of frames in the current segment
        n_frames = max(int(len(segment) / frame_length_samples), 1)
        
        # Calculate the energy for each frame in the segment
        energy = np.array([np.sum(segment[j * frame_length_samples:(j + 1) * frame_length_samples] ** 2) for j in range(n_frames)])
        
        # Calculate the Zero-Crossing Rate for each frame in the segment
        zcr = np.array([np.sum(librosa.zero_crossings(segment[j * frame_length_samples:(j + 1) * frame_length_samples], pad=False)) for j in range(n_frames)])
        
        # Add the energy threshold to the array
        # The threshold is the mean plus the standard deviation of the energy in the segment
        energy_thresholds.append(np.mean(energy) + np.std(energy))
        
        # Add the ZCR threshold to the array
        # The threshold is the mean plus the standard deviation of ZCR in the segment
        zcr_thresholds.append(np.mean(zcr) + np.std(zcr))
    
    # Return the lists of thresholds
    return energy_thresholds, zcr_thresholds




def analyze_chunks(audio, sr, min_chunk_length_samples, max_chunk_length_samples, frame_length_samples, overlap_samples):
    """
    Analyzes the audio to find chunks based on energy and Zero-Crossing Rate (ZCR).

    This function divides the audio into frames and determines whether each frame is silent or active.
    Active frames are aggregated into chunks. Silent frames signal the end of a chunk,
    taking into account the specified minimum and maximum chunk lengths.
    """
    # Dividing the audio into five segments for threshold calculation
    segment_length_samples = len(audio) // 5
    # Compute dynamic thresholds for energy and ZCR for each segment
    energy_thresholds, zcr_thresholds = compute_dynamic_thresholds(audio, sr, frame_length_samples, segment_length_samples)

    chunks = []  # Initialization of the list for the found chunks
    current_chunk_start = None  # Starting point of the current chunk
    last_valid_end = None  # Endpoint of the last valid frame in the current chunk
    is_previous_frame_silent = False  # State of the previous frame

    # Iterating over the audio in frame steps
    for i in range(0, len(audio), frame_length_samples):
        segment_index = i // segment_length_samples
        segment_index = min(segment_index, len(energy_thresholds) - 1)  # Prevent index overflow

        # Assigning thresholds for the current frame
        energy_threshold = energy_thresholds[segment_index]
        zcr_threshold = zcr_thresholds[segment_index]

        # Extracting the current frame
        frame = audio[i:min(i + frame_length_samples, len(audio))]
        # Calculating energy and ZCR for the frame
        frame_energy = np.sum(frame ** 2)
        frame_zcr = np.sum(librosa.zero_crossings(frame, pad=False))

        # Determining whether the frame is silent
        is_silent = frame_energy <= energy_threshold and frame_zcr <= zcr_threshold

        # If no chunk has started and the frame is not silent, start a new chunk
        if current_chunk_start is None and not is_silent:
            current_chunk_start = i  # Start of a new chunk
            last_valid_end = i + frame_length_samples

        # If a chunk is active
        if current_chunk_start is not None:
            # If the frame is silent or the maximum chunk length has been reached
            if is_silent and (is_previous_frame_silent or i + frame_length_samples - current_chunk_start >= max_chunk_length_samples):
                # Check if the chunk is long enough to be saved
                if last_valid_end and last_valid_end - current_chunk_start >= min_chunk_length_samples:
                    chunks.append((current_chunk_start, last_valid_end + overlap_samples))  # Save with overlap
                    current_chunk_start = None
            else:
                # Update the end of the current chunk
                last_valid_end = i + frame_length_samples
                # If the frame is not silent and the maximum chunk length has been reached
                if not is_silent and i + frame_length_samples - current_chunk_start >= max_chunk_length_samples:
                    chunks.append((current_chunk_start, min(len(audio), last_valid_end + overlap_samples)))  # Save with overlap
                    current_chunk_start = None

        is_previous_frame_silent = is_silent

    # Saving the last chunk, if necessary
    if current_chunk_start is not None and last_valid_end - current_chunk_start >= min_chunk_length_samples:
        chunks.append((current_chunk_start, min(len(audio), last_valid_end + overlap_samples)))  # Save with overlap

    return chunks