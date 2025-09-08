#!/usr/bin/env python3
"""
Test script for Hindi language support in MediBot
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'SRC'))

from helper import detect_language, preprocess_hindi_text

def test_language_detection():
    """Test language detection functionality"""
    
    # Test cases
    test_cases = [
        ("Hello, how are you?", "en"),
        ("What is diabetes?", "en"),
        ("नमस्ते, आप कैसे हैं?", "hi"),
        ("मधुमेह क्या है?", "hi"),
        ("डायबिटीज के लक्षण क्या हैं?", "hi"),
        ("What are the symptoms of diabetes?", "en"),
        ("Hi", "en"),  # Short text
        ("नमस्ते", "hi"),  # Short Hindi text
    ]
    
    print("Testing Language Detection:")
    print("=" * 50)
    
    for text, expected in test_cases:
        detected = detect_language(text)
        status = "✓" if detected == expected else "✗"
        print(f"{status} '{text}' -> Expected: {expected}, Got: {detected}")
    
    print("\nTesting Hindi Text Preprocessing:")
    print("=" * 50)
    
    hindi_texts = [
        "मधुमेह   के   लक्षण   क्या   हैं?",
        "  नमस्ते  ",
        "डायबिटीज\t\t\tके\t\t\tकारण",
    ]
    
    for text in hindi_texts:
        processed = preprocess_hindi_text(text)
        print(f"Original: '{text}'")
        print(f"Processed: '{processed}'")
        print()

if __name__ == "__main__":
    test_language_detection()
