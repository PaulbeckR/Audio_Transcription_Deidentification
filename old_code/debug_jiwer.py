from jiwer import process_words

output = process_words("hello world", "hello there world")
print("Type:", type(output.alignments))
print("First item:", output.alignments[0] if output.alignments else "empty")
print("First item type:", type(output.alignments[0]) if output.alignments else "N/A")

# Try to access attributes
if output.alignments:
    first = output.alignments[0]
    if hasattr(first, 'type'):
        print("Has .type attribute")
    elif isinstance(first, list):
        print("Is a list:", first)
    else:
        print("Object:", first)
        print("Dir:", [x for x in dir(first) if not x.startswith('_')])