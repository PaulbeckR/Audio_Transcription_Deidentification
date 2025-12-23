# Speaker Segmentation and Diarization (identify and label speakers)

## Reference: Recent Methods in and Algorithms in Speech Segmentation Tasks: link.springer.com/chapter/10.1007/978-3-031-70259-4_21

1. Process:
    a. Feature vector extraction from the input speech 
    b. Speech activity detection, speech classified into speech and non-speech data - determining speaker 
    c. Segmentatio nmethod : speaker changes 
    d. Clustering method, creating clusters of speakers. 

2. Implementation Algo:
    a. Data Prep: WAV 
    b. Converstion to single channel: consistent processing
    c. Speech Activity detection: segments identified and saved in WAV
    d. Diarization with speaker recognition: all audio checked with specific speaker, moved to same folder 
    e. MAP Model Adaptation: (Max a Posteriori) on all audio recordings, yielding shift vectors.  
    f. Clustering of shift vectors: shift vectors into clustering function. Groups by similarity and identification of speakers. 


3. Models for Feature Extraction:
    a. Short segments 20-40ms. Signal transformations > Mel-Frequency Cepstral Coefficients (MFCC) method. 

4. Speech Activity Detection: noise/clicks/speech
    a. GMM-speech vs non-speech
        - dependent on training data: poor adaptation to changes in recording conditions
            - overcome: training data > signal energy-based classification. More flexible. more robust. 

5. Speaker changes: 
    a. Feature characteristics, ML methods, DNN, 
    b. abrupt change; timbre, intonation, etc. 
    c. Enhanced: post-processing : smoothing, outlier filtering, utilization of contextual information

6. Clustering: grouping by speaker
    methods:
        a. Hierarchical clustering; k-means, 
            - k-means: predefine number of clusters
        b. DBSCAN - auto select cluster number
        c. optimal cluster selection using elbow method
    
    b. for each pair of feature vectors, similarity measure is calculated, Euclidean distance, cosine, or others. 
    c. Post-processing refining boundaries, 

7. Frameworks: 
    a. pyAudioAnalysis: feature extraction, classification, segmentation, can include speaker diarization
    b. Pyannote: most utilized, pre-trained, may need retuning for mixed-quality audio
    c. Kaldi, research and industry
    d. pyAudioDiarization: ML techniques, 
    c. NVIDIA NeMO: 


# Model Evaluation: 

1. Metrics

    a. Diarization: Diarization Error Rate: 
        DER = E_spkr + E_fa + E_miss + E_ovl 
        E_spkr : error of misclassifying a speaker 
        E_fa: false alarm error, percentage of time allocated to speakers that should have been 'non-speech'
        E_miss: missed speech error, percentage of time speech was identified as 'non-speech'
        E_ovl: overlapped speaker error, percentage of time when some speakers in a segment where not assigned ot any specific speaker. 

    
2. 

