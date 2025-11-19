#!/usr/bin/env python
import json
import sys

json_file = sys.argv[1]

with open(json_file) as f:
    data = json.load(f)

segments = data['segments']

# Count segments with backfill
backfilled_segments = [s for s in segments if s.get('contains_backfill')]
segments_with_sv = [s for s in backfilled_segments if 'sensevoice_text' in s]

# Count backfilled words
total_backfilled_words = 0
words_with_sv = 0

for seg in segments:
    for word in seg.get('words', []):
        if word.get('is_backfill'):
            total_backfilled_words += 1
            if 'sensevoice_text' in word:
                words_with_sv += 1

print(f"📊 SenseVoice Integration Results:")
print(f"\nSegments:")
print(f"  Total segments: {len(segments)}")
print(f"  Segments with backfill: {len(backfilled_segments)}")
print(f"  Segments with SenseVoice data: {len(segments_with_sv)} ({100*len(segments_with_sv)/len(backfilled_segments):.1f}%)")

print(f"\nWords:")
print(f"  Total backfilled words: {total_backfilled_words}")
print(f"  Backfilled words with SenseVoice: {words_with_sv} ({100*words_with_sv/total_backfilled_words if total_backfilled_words > 0 else 0:.1f}%)")

# Show some examples
print(f"\n📝 Sample SenseVoice Results:")
count = 0
for seg in segments:
    for word in seg.get('words', []):
        if word.get('is_backfill') and 'sensevoice_text' in word:
            whisper_text = word.get('text', word.get('word', ''))
            sv_text = word.get('sensevoice_text', '')
            emotion = word.get('sensevoice_emotion')
            event = word.get('sensevoice_event')
            
            print(f"\n  Example {count+1}:")
            print(f"    Whisper: '{whisper_text}'")
            print(f"    SenseVoice: '{sv_text}'")
            if emotion:
                print(f"    Emotion: {emotion}")
            if event:
                print(f"    Event: {event}")
            
            count += 1
            if count >= 5:
                break
    if count >= 5:
        break
