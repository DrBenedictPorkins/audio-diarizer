#!/usr/bin/env python3

from src.audio_diarizer.audio_processor import AudioProcessor

def test_preprocessing():
    processor = AudioProcessor()
    try:
        result = processor.preprocess_audio('uploads/a3c850cf-0cf9-4695-9c05-cfc7318c3505_test.m4a')
        print('Success:', result)
        return result
    except Exception as e:
        print('Error:', e)
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    test_preprocessing()