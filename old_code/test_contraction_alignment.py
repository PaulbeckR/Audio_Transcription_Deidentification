"""
Test how jiwer handles contraction vs expanded form alignment
"""
from jiwer import process_words

print("="*80)
print("TESTING CONTRACTION ALIGNMENT IN JIWER")
print("="*80)

# Test 1: Simple contraction difference
print("\nTest 1: it's vs it is")
print("-" * 40)
ref1 = "it's raining today"
hyp1 = "it is raining today"

output1 = process_words(ref1, hyp1)
print(f"Reference:  '{ref1}'  ({len(ref1.split())} words)")
print(f"Hypothesis: '{hyp1}' ({len(hyp1.split())} words)")
print(f"\nWER: {output1.wer:.2%}")
print(f"Substitutions: {output1.substitutions}")
print(f"Deletions:     {output1.deletions}")
print(f"Insertions:    {output1.insertions}")
print(f"Hits:          {output1.hits}")

# Test 2: Your example - dog's vs dog is
print("\n" + "="*80)
print("\nTest 2: The dog's hungry vs The dog is hungry")
print("-" * 40)
ref2 = "The dog's hungry"
hyp2 = "The dog is hungry"

output2 = process_words(ref2, hyp2)
print(f"Reference:  '{ref2}'  ({len(ref2.split())} words)")
print(f"Hypothesis: '{hyp2}' ({len(hyp2.split())} words)")
print(f"\nWER: {output2.wer:.2%}")
print(f"Substitutions: {output2.substitutions}")
print(f"Deletions:     {output2.deletions}")
print(f"Insertions:    {output2.insertions}")
print(f"Hits:          {output2.hits}")

# Show alignment
print(f"\nAlignment visualization:")
print(f"  REF: {' | '.join(ref2.split())}")
print(f"  HYP: {' | '.join(hyp2.split())}")

# Test 3: Multiple contractions
print("\n" + "="*80)
print("\nTest 3: Multiple contractions in sentence")
print("-" * 40)
ref3 = "I'm sure it's fine we'll see"
hyp3 = "I am sure it is fine we will see"

output3 = process_words(ref3, hyp3)
print(f"Reference:  '{ref3}'  ({len(ref3.split())} words)")
print(f"Hypothesis: '{hyp3}' ({len(hyp3.split())} words)")
print(f"\nWER: {output3.wer:.2%}")
print(f"Substitutions: {output3.substitutions}")
print(f"Deletions:     {output3.deletions}")
print(f"Insertions:    {output3.insertions}")
print(f"Hits:          {output3.hits}")

# Test 4: Reverse - expanded in reference, contraction in hypothesis
print("\n" + "="*80)
print("\nTest 4: REVERSE - it is vs it's")
print("-" * 40)
ref4 = "it is raining today"
hyp4 = "it's raining today"

output4 = process_words(ref4, hyp4)
print(f"Reference:  '{ref4}' ({len(ref4.split())} words)")
print(f"Hypothesis: '{hyp4}'  ({len(hyp4.split())} words)")
print(f"\nWER: {output4.wer:.2%}")
print(f"Substitutions: {output4.substitutions}")
print(f"Deletions:     {output4.deletions}")
print(f"Insertions:    {output4.insertions}")
print(f"Hits:          {output4.hits}")

print("\n" + "="*80)
print("\nCONCLUSION:")
print("-" * 40)
print("jiwer uses dynamic programming (Levenshtein distance) to find optimal")
print("word alignment, so it DOES handle contractions intelligently by finding")
print("the alignment that minimizes total errors.")
print("="*80)