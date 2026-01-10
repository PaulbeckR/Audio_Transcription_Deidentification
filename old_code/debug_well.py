"""
Debug the we'll normalization issue
"""
from src.metrics import calculate_wer
import re

ref = "I'm sure it's fine we'll see"
hyp = "I am sure it is fine we will see"

# Test what the normalization does
print("Original:")
print(f"  REF: '{ref}'")
print(f"  HYP: '{hyp}'")

# Manual step-by-step normalization to debug
ref_test = ref
hyp_test = hyp

# Step 1: Expand contractions
contractions = {
    r"\bi'm\b": "i am",
    r"\bit's\b": "it is",
    r"\bwe'll\b": "we will",
}

for contraction, expansion in contractions.items():
    ref_test = re.sub(contraction, expansion, ref_test, flags=re.IGNORECASE)
    hyp_test = re.sub(contraction, expansion, hyp_test, flags=re.IGNORECASE)

print("\nAfter contraction expansion:")
print(f"  REF: '{ref_test}'")
print(f"  HYP: '{hyp_test}'")

# Step 2: Expand remaining 's
ref_test = re.sub(r"(\w)'s\b", r"\1 is", ref_test, flags=re.IGNORECASE)
hyp_test = re.sub(r"(\w)'s\b", r"\1 is", hyp_test, flags=re.IGNORECASE)

print("\nAfter 's expansion:")
print(f"  REF: '{ref_test}'")
print(f"  HYP: '{hyp_test}'")

# Check if they match
print("\nDo they match?", ref_test.lower() == hyp_test.lower())

# Now test with actual function
result = calculate_wer(ref, hyp)
print("\n" + "="*60)
print("ACTUAL WER RESULT:")
print(f"WER: {result['wer']:.2%}")
print(f"Subs: {result['substitutions']}, Ins: {result['insertions']}, Dels: {result['deletions']}, Hits: {result['hits']}")