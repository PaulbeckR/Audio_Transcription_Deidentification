# Start of ReadMe

#Important packages:

import whisper
from pyannote.audio import Pipeline
import csv
import ffmpeg
import pydub
from pydub import AudioSegment
import os
import torch

# Project: 


The goal of this project is to process audio files into clean transcripts with high accuracy. The project uses pyannote audio and WhisperAI. The primary code file for processing an audio file into a transcript is in transcript_notebook.ipynb. Audio files are read and segmented by speaker diarization using pyannote. Audio files are separated into chunks for processing through WhisperAI. 

The output is low and has a high error rate. 

The project was imported from a previous computer, direct folder locations will need to change. 

I need help with this project to

 1. Increase accuracy. The current machine has a GPU available. Other methods may be helpful for processing. 

2. Build out pre-training module: I have several already transcribed audio files. They are not time-stamped, and time-stamps are needed for RTTM formatting and parsign the audio files correctly. There may be other methods to work around this. 

3. Increase speed: Also tied to accuracy and GPU. 

4. Build out metrics for tracking accuracy and performance. 

5. Organize and document codebase better. Can incorporate additional tools as needed. 

6. The final product should be able to auto transcribe files in a given folder, saving transcripts to an adjacent folder. 


Needs: 

Audio files and their transcripts in the primary project folders are not to be read/uploaded by any cloud-based resources. All local. This includes claude. Hamelet_test is a safe test to use/read. 

Claude is not allowed to access the Box folder on this computer unless only for filename access and local memory only is guaranteed. 


Problems: 

Audio transcriptoin is slow and full of errors. Need to increase speed, accuracy, and automation. 

My set of already transcribed files are not time-stamped and need to be in order to use them for pretraining data. 



Ideas: 

Pretraining: Once I have an initial fully time-stamped transcript that is accurate. I can load that into the pre-training library to support whisper and pyannote accuracy. With each additional transcription complete, this can be added to the pre-training dataset to support enhanced training. This addition can stop around an idea number to balance speed/performance. 


Later on - can incorporate SpaCy and NLTK for auto-deidentificaton and flagging of PHI 


Audio Context: 

Audio files are long-form interviews between two people. There are only ever two people (except for hamlet test). 

There are three total interviewers across all audio files.The second party differs greatly, with each individual interviewed 1-3 times by a single interviewer. Pretraining data should include a balanced mixed of each interviewer. 

Audio files include a label like S1, S2, S3 for "sessions". Each session includes a specific curriculum/topic. This means that there should be a moderate to high degree of overlap in words used by the interviewers (and potentially interviewees) across each session. This is why adding additional pretraining data can be advantageous. Words overlap and voice characteristics overlap. 




