"""
Validate that the normalization function works correctly
"""
from src.metrics import calculate_wer

print("=" * 80)
print("VALIDATION: Contraction & Possessive Normalization")
print("=" * 80)

# Test 1: Possessive handling
print("\nTest 1: Possessive (should be 0% WER after normalization)")
print("-" * 40)
ref1 = "The dog's hungry"
hyp1 = "The dog is hungry"
result1 = calculate_wer(ref1, hyp1)
print(f"Reference:  '{ref1}'")
print(f"Hypothesis: '{hyp1}'")
print(f"WER: {result1['wer']:.2%}")
print(f"Subs: {result1['substitutions']}, Ins: {result1['insertions']}, Dels: {result1['deletions']}, Hits: {result1['hits']}")

# Test 2: Contraction handling
print("\n" + "=" * 80)
print("\nTest 2: Contraction (should be 0% WER after normalization)")
print("-" * 40)
ref2 = "it's raining today"
hyp2 = "it is raining today"
result2 = calculate_wer(ref2, hyp2)
print(f"Reference:  '{ref2}'")
print(f"Hypothesis: '{hyp2}'")
print(f"WER: {result2['wer']:.2%}")
print(f"Subs: {result2['substitutions']}, Ins: {result2['insertions']}, Dels: {result2['deletions']}, Hits: {result2['hits']}")

# Test 3: Multiple contractions
print("\n" + "=" * 80)
print("\nTest 3: Multiple contractions (should be 0% WER)")
print("-" * 40)
ref3 = "I'm sure it's fine we'll see"
hyp3 = "I am sure it is fine we will see"
result3 = calculate_wer(ref3, hyp3)
print(f"Reference:  '{ref3}'")
print(f"Hypothesis: '{hyp3}'")
print(f"WER: {result3['wer']:.2%}")
print(f"Subs: {result3['substitutions']}, Ins: {result3['insertions']}, Dels: {result3['deletions']}, Hits: {result3['hits']}")

# Test 4: Reverse - contraction in hypothesis
print("\n" + "=" * 80)
print("\nTest 4: REVERSE - contraction in hypothesis (should be 0% WER)")
print("-" * 40)
ref4 = "it is raining today"
hyp4 = "it's raining today"
result4 = calculate_wer(ref4, hyp4)
print(f"Reference:  '{ref4}'")
print(f"Hypothesis: '{hyp4}'")
print(f"WER: {result4['wer']:.2%}")
print(f"Subs: {result4['substitutions']}, Ins: {result4['insertions']}, Dels: {result4['deletions']}, Hits: {result4['hits']}")

# Test 5: Linda's possessive (should match actual name)
print("\n" + "=" * 80)
print("\nTest 5: Proper name possessive (should be 0% WER)")
print("-" * 40)
ref5 = "Linda's 74 and she's still mentally sharp"
hyp5 = "Linda is 74 and she is still mentally sharp"
result5 = calculate_wer(ref5, hyp5)
print(f"Reference:  '{ref5}'")
print(f"Hypothesis: '{hyp5}'")
print(f"WER: {result5['wer']:.2%}")
print(f"Subs: {result5['substitutions']}, Ins: {result5['insertions']}, Dels: {result5['deletions']}, Hits: {result5['hits']}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("All tests should show 0% WER if normalization is working correctly")
print("=" * 80)