"""
Analyze contraction differences between ground truth and WhisperX output
"""
import pandas as pd
import re
from collections import Counter

# Load both files
gt_df = pd.read_csv('Audio_Local_tests/transcription_test/gpt_test_truth.csv')
wx_df = pd.read_csv('Audio_Local_tests/baseline_output/pretrain_test/gpt_test_wx-lv2/gpt_test_wx-lv2_large-v2_cuda.csv')

# Extract text
gt_text = ' '.join(gt_df['Transcription'].dropna().astype(str))
wx_text = ' '.join(wx_df['Transcription'].dropna().astype(str))

# Remove punctuation (except apostrophes)
punctuation_pattern = r'[,\.;:!?\"\(\)\[\]\{\}\-—–…]'
gt_clean = re.sub(punctuation_pattern, '', gt_text)
wx_clean = re.sub(punctuation_pattern, '', wx_text)
gt_clean = re.sub(r'\s+', ' ', gt_clean).strip()
wx_clean = re.sub(r'\s+', ' ', wx_clean).strip()

# Find all contractions in both texts
contractions_pattern = r"\b\w+'\w+\b"
gt_contractions = re.findall(contractions_pattern, gt_clean, re.IGNORECASE)
wx_contractions = re.findall(contractions_pattern, wx_clean, re.IGNORECASE)

print('=== CONTRACTIONS FOUND ===')
print(f'Ground Truth contractions: {len(gt_contractions)}')
print(f'WhisperX contractions: {len(wx_contractions)}')
print()

# Count unique contractions
gt_counter = Counter([c.lower() for c in gt_contractions])
wx_counter = Counter([c.lower() for c in wx_contractions])

print('Ground Truth unique contractions:')
for contraction, count in sorted(gt_counter.items(), key=lambda x: -x[1]):
    print(f'  {contraction}: {count}')

print()
print('WhisperX unique contractions:')
for contraction, count in sorted(wx_counter.items(), key=lambda x: -x[1]):
    print(f'  {contraction}: {count}')

print()
print('Differences in contraction counts:')
all_contractions = set(gt_counter.keys()) | set(wx_counter.keys())
diff_count = 0
for c in sorted(all_contractions):
    gt_count = gt_counter.get(c, 0)
    wx_count = wx_counter.get(c, 0)
    diff = wx_count - gt_count
    if diff != 0:
        diff_count += abs(diff)
        sign = '+' if diff > 0 else ''
        print(f'  {c}: GT={gt_count}, WX={wx_count} (diff: {sign}{diff})')

print()
print(f'Total contraction count differences: {diff_count}')