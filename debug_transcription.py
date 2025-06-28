from src.audio_diarizer.audio_processor import AudioProcessor
from src.audio_diarizer.transcription import SpeechTranscriber

# Use the segments we got from diarization
segments = [(0.03096875, 6.949718750000001, 'Speaker A'), 
            (6.949718750000001, 21.96846875, 'Speaker B'), 
            (22.54221875, 44.17596875, 'Speaker C')]

try:
    processor = AudioProcessor()
    transcriber = SpeechTranscriber()
    
    # Load audio
    print("Loading audio...")
    audio, sr = processor.load_audio('uploads/processed_a3c850cf-0cf9-4695-9c05-cfc7318c3505_test.m4a')
    print(f"Audio loaded: {len(audio)} samples at {sr}Hz")
    
    # Segment audio
    print("Segmenting audio...")
    audio_segments = processor.segment_audio(audio, segments, sr)
    print(f"Created {len(audio_segments)} audio segments")
    
    # Transcribe segments
    print("Transcribing segments...")
    transcribed_segments = transcriber.transcribe_segments(audio_segments)
    print("Transcription success:", len(transcribed_segments), "segments")
    
    for i, segment in enumerate(transcribed_segments[:3]):
        print(f"Segment {i}: {segment}")
        
except Exception as e:
    print('Transcription error:', e)
    import traceback
    traceback.print_exc()