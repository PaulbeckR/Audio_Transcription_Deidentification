from src.utils import convert_to_wav

input1 = "Audio_Local_tests/audio_files/gpt_test2.m4a"
output1 = "Audio_Local_tests/audio_files/gpt_test2.wav"

input2 = "Audio_Local_tests/audio_files/gpt_test.m4a"
output2 = "Audio_Local_tests/audio_files/gpt_test.wav"

convert_to_wav(input1, output1)

convert_to_wav(input2, output2)