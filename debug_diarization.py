from src.audio_diarizer.diarization import SpeakerDiarizer

diarizer = SpeakerDiarizer()
try:
    segments = diarizer.diarize('uploads/processed_a3c850cf-0cf9-4695-9c05-cfc7318c3505_test.m4a')
    print('Diarization success:', len(segments), 'segments')
    print('First few segments:', segments[:3] if segments else 'None')
except Exception as e:
    print('Diarization error:', e)
    import traceback
    traceback.print_exc()