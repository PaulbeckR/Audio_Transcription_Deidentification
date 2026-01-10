# Edit truth transcript and resave - speaker count. 


import pandas as pd
import numpy as np
import os 
import openpyxl

test_path = 'C://Users//paulbeck//Documents//Audio_Local_tests//transcription_test//hamlet_test_truth.xlsx'
test_fold = 'C://Users//paulbeck//Documents//Audio_Local_tests//transcription_test'


df = pd.read_excel(test_path)

speaker_col = df['Speaker']
speaker_col = speaker_col.str.strip()

print("before", speaker_col)

speaker_col.replace(['Nan', 'None', '' , 'nan'], np.nan, inplace = True)

unique_speakers = speaker_col.dropna().unique()

#unique_speakers, speaker_ids = pd.factorize(speaker_col)

print(unique_speakers)


speaker_mapping = {name: f'SPEAKER_{i+1:02}' for i , name in enumerate(unique_speakers)}


df['Speaker'] = speaker_col.map(speaker_mapping)

print(df['Speaker'])

df.to_excel( test_fold + 'updated_speakers.xlsx', index=False)