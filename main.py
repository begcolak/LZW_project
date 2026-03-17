from LZW import LZWCoding
from pathlib import Path

def run_test_suite():
    Path("input").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

    print("--- 1. Testing Text Compression ---")
    text_proc = LZWCoding("sample")
    text_proc.compress_text_file()
    text_proc.decompress_text_file()

    print("\n--- 2. Testing Grayscale Compression ---")
    gray_proc = LZWCoding("thumbs_up")
    gray_proc.compress_Grayscale()
    gray_proc.decompress_Grayscale()

    print("\n--- 3. Testing Grayscale Diff Compression ---")
    diff_proc = LZWCoding("thumbs_up")
    diff_proc.compress_Grayscale_Diff()
    diff_proc.decompress_Grayscale_Diff()

    print("\n--- 4. Testing RGB Compression ---")
    rgb_proc = LZWCoding("thumbs_up")
    rgb_proc.compress_RGB()
    rgb_proc.decompress_RGB()

    print("\n--- 5. Testing RGB Diff Compression ---")
    rgb_diff_proc = LZWCoding("thumbs_up")
    rgb_diff_proc.compress_RGB_Diff()
    rgb_diff_proc.decompress_RGB_Diff()

    print("\nAll tests finished successfully.")

if __name__ == "__main__":
    run_test_suite()