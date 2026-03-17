import math
import os
from pathlib import Path
from io import StringIO
import numpy as np
from basic_image_operations import *

class LZWCoding:
    def __init__(self, filename):
        self.filename = filename

        # --- PATH CONFIGURATION ---
        # Locates the actual directory where the script is residing
        self.base_dir = Path(__file__).parent.absolute()

        # Set directories relative to the base directory
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output"

        # Create output directory if it does not exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_metrics(self, original_size, compressed_size):
        """
        Calculates CR (Compression Ratio), CF (Compression Factor),
        and SS (Space Saving) metrics.
        """
        if original_size == 0 or compressed_size == 0:
            return 0, 0, 0

        cr = original_size / compressed_size
        cf = (compressed_size / original_size) * 100
        ss = (1 - (compressed_size / original_size)) * 100

        return round(cr, 2), round(cf, 2), round(ss, 2)

    def calculate_entropy(self, data_stream):
        """Calculates Shannon Entropy to measure data complexity."""
        if len(data_stream) == 0: return 0
        _, counts = np.unique(data_stream, return_counts=True)
        probs = counts / len(data_stream)
        return -np.sum(probs * np.log2(probs))

    def _format_text(self, text):
        """Escapes special characters for safe CSV logging."""
        if not text: return ""
        return str(text).replace('\n', '\\n').replace('\r', '\\r').replace("'", "\\'")

    def save_log_csv(self, logs, suffix):
        """Saves the step-by-step LZW process to a CSV file."""
        output_path = self.output_dir / f"{self.filename}_{suffix}.csv"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("w,k,Output,Index,Symbol\n")
            for entry in logs:
                f.write(f"{entry.get('w','')},{entry.get('k','')},{entry.get('Output','')},"
                        f"{entry.get('Index','')},{entry.get('Symbol','')}\n")

    def pack_binary(self, bit_tuples):
        """Packs dynamic-length codes (9, 10, 11 bits...) into 8-bit bytes."""
        bit_string = ""
        for value, bit_length in bit_tuples:
            bit_string += bin(value)[2:].zfill(bit_length)

        padding_size = (8 - len(bit_string) % 8) % 8
        header = f"{padding_size:08b}"
        final_bits = header + bit_string + ("0" * padding_size)

        return bytearray(int(final_bits[i:i+8], 2) for i in range(0, len(final_bits), 8))

    def unpack_binary(self, binary_data, is_diff=False):
        """Unpacks byte data back into LZW codes using dynamic bit growth logic."""
        if not binary_data: return []
        all_bits = "".join(f"{b:08b}" for b in binary_data)

        padding_len = int(all_bits[:8], 2)
        payload = all_bits[8:-padding_len] if padding_len > 0 else all_bits[8:]

        extracted_codes = []
        dict_size = 512 if is_diff else 256
        current_bit_len = 9
        pointer = 0

        if len(payload) < current_bit_len: return []
        extracted_codes.append(int(payload[:current_bit_len], 2))
        pointer = current_bit_len
        dict_size += 1

        while pointer < len(payload):
            if dict_size > (1 << current_bit_len):
                current_bit_len += 1
            if pointer + current_bit_len > len(payload): break
            code = int(payload[pointer : pointer + current_bit_len], 2)
            extracted_codes.append(code)
            pointer += current_bit_len
            dict_size += 1
        return extracted_codes

    def encode(self, sequence, is_diff=False):
        """Core LZW Encoding algorithm."""
        init_size = 512 if is_diff else 256
        dictionary = {chr(i): i for i in range(init_size)}
        w = ""
        compressed_output = []
        process_logs = []
        bit_width = 9
        next_available_index = init_size

        for k in sequence:
            wk = w + k
            if wk in dictionary:
                w = wk
            else:
                if w != "":
                    compressed_output.append((dictionary[w], bit_width))
                    process_logs.append({
                        "w": self._format_text(w), "k": self._format_text(k),
                        "Output": dictionary[w], "Index": next_available_index, "Symbol": self._format_text(wk)
                    })
                dictionary[wk] = next_available_index
                next_available_index += 1
                if next_available_index > (1 << bit_width):
                    bit_width += 1
                w = k
        if w:
            compressed_output.append((dictionary[w], bit_width))
            process_logs.append({"w": self._format_text(w), "k": "", "Output": dictionary[w], "Index": "", "Symbol": ""})
        return compressed_output, process_logs

    def decode(self, codes, is_diff=False):
        """Core LZW Decoding algorithm."""
        init_size = 512 if is_diff else 256
        dictionary = {i: chr(i) for i in range(init_size)}
        output_buffer = StringIO()
        step_logs = []
        next_available_index = init_size
        if not codes: return "", []
        code_list = list(codes)
        previous_code = code_list.pop(0)
        w = dictionary[previous_code]
        output_buffer.write(w)
        for current_code in code_list:
            if current_code in dictionary:
                entry = dictionary[current_code]
            elif current_code == next_available_index:
                entry = w + w[0]
            else:
                raise ValueError("Decompression error!")
            output_buffer.write(entry)
            step_logs.append({"w": self._format_text(w), "k": current_code, "Output": self._format_text(entry), "Index": next_available_index, "Symbol": self._format_text(w + entry[0])})
            dictionary[next_available_index] = w + entry[0]
            next_available_index += 1
            w = entry
        return output_buffer.getvalue(), step_logs

    # --- Compression and Decompression Methods ---

    def compress_text_file(self):
        with open(self.input_dir / f"{self.filename}.txt", 'r', encoding='utf-8') as f:
            content = f.read()
        codes, logs = self.encode(content)
        self.save_log_csv(logs, "text_encode_log")
        with open(self.output_dir / f"{self.filename}_text_compressed.bin", 'wb') as f:
            f.write(self.pack_binary(codes))

    def decompress_text_file(self):
        with open(self.output_dir / f"{self.filename}_text_compressed.bin", 'rb') as f:
            codes = self.unpack_binary(f.read())
        content, logs = self.decode(codes)
        self.save_log_csv(logs, "text_decode_log")
        with open(self.output_dir / f"{self.filename}_text_restored.txt", 'w', encoding='utf-8') as f:
            f.write(content)

    def compress_Grayscale(self):
        img_path = self.input_dir / f"{self.filename}.bmp"
        img = read_image_from_file(str(img_path))
        gray_img = color_to_gray(img)
        pixels = PIL_to_np(gray_img)
        rows, cols = pixels.shape
        codes, logs = self.encode("".join(chr(p) for p in pixels.flatten()))
        self.save_log_csv(logs, "gray_encode_log")
        output_file = self.output_dir / f"{self.filename}_gray.bin"
        with open(output_file, 'wb') as f:
            f.write(rows.to_bytes(4, 'big') + cols.to_bytes(4, 'big') + self.pack_binary(codes))

    def decompress_Grayscale(self):
        with open(self.output_dir / f"{self.filename}_gray.bin", 'rb') as f:
            data = f.read()
            rows, cols = int.from_bytes(data[:4], 'big'), int.from_bytes(data[4:8], 'big')
            codes = self.unpack_binary(data[8:])
        decoded_str, logs = self.decode(codes)
        self.save_log_csv(logs, "gray_decode_log")
        reconstructed = np.array([ord(x) for x in decoded_str]).reshape(rows, cols).astype(np.uint8)
        write_image_to_file(np_to_PIL(reconstructed), str(self.output_dir / f"{self.filename}_gray_restored.bmp"))

    def compress_Grayscale_Diff(self):
        img_path = self.input_dir / f"{self.filename}.bmp"
        img = read_image_from_file(str(img_path))
        gray = color_to_gray(img)
        flat_pixels = PIL_to_np(gray).flatten().astype(np.int16)
        rows, cols = PIL_to_np(gray).shape
        diff_stream = np.zeros_like(flat_pixels)
        diff_stream[0] = flat_pixels[0]
        for i in range(1, len(flat_pixels)): diff_stream[i] = flat_pixels[i] - flat_pixels[i-1]
        write_image_to_file(np_to_PIL(normalize_diff_image(diff_stream.reshape(rows, cols))), str(self.output_dir / f"{self.filename}_gray_diff_visual.bmp"))
        codes, logs = self.encode("".join(chr(int(d) + 255) for d in diff_stream), is_diff=True)
        self.save_log_csv(logs, "gray_diff_encode_log")
        with open(self.output_dir / f"{self.filename}_gray_diff.bin", 'wb') as f:
            f.write(rows.to_bytes(4, 'big') + cols.to_bytes(4, 'big') + self.pack_binary(codes))

    def decompress_Grayscale_Diff(self):
        with open(self.output_dir / f"{self.filename}_gray_diff.bin", 'rb') as f:
            data = f.read()
            rows, cols = int.from_bytes(data[:4], 'big'), int.from_bytes(data[4:8], 'big')
            codes = self.unpack_binary(data[8:], is_diff=True)
        decoded_str, logs = self.decode(codes, is_diff=True)
        self.save_log_csv(logs, "gray_diff_decode_log")
        diffs = np.array([ord(x) - 255 for x in decoded_str])
        restored = np.zeros_like(diffs)
        restored[0] = diffs[0]
        for i in range(1, len(diffs)): restored[i] = restored[i-1] + diffs[i]
        final_img = restored.reshape(rows, cols).clip(0, 255).astype(np.uint8)
        write_image_to_file(np_to_PIL(final_img), str(self.output_dir / f"{self.filename}_gray_diff_restored.bmp"))

    def compress_RGB(self):
        img = read_image_from_file(str(self.input_dir / f"{self.filename}.bmp"))
        arr = PIL_to_np(img)
        rows, cols, _ = arr.shape
        codes, logs = self.encode("".join(chr(p) for p in arr.flatten()))
        self.save_log_csv(logs, "rgb_encode_log")
        with open(self.output_dir / f"{self.filename}_rgb.bin", 'wb') as f:
            f.write(rows.to_bytes(4, 'big') + cols.to_bytes(4, 'big') + self.pack_binary(codes))

    def decompress_RGB(self):
        with open(self.output_dir / f"{self.filename}_rgb.bin", 'rb') as f:
            data = f.read()
            rows, cols = int.from_bytes(data[:4], 'big'), int.from_bytes(data[4:8], 'big')
            codes = self.unpack_binary(data[8:])
        decoded_str, logs = self.decode(codes)
        self.save_log_csv(logs, "rgb_decode_log")
        pixels = np.array([ord(x) for x in decoded_str]).reshape(rows, cols, 3).astype(np.uint8)
        write_image_to_file(np_to_PIL(pixels), str(self.output_dir / f"{self.filename}_rgb_restored.bmp"))

    def compress_RGB_Diff(self):
        img = read_image_from_file(str(self.input_dir / f"{self.filename}.bmp"))
        arr = PIL_to_np(img).astype(np.int16)
        rows, cols, _ = arr.shape
        diff_channels = []
        visuals = []
        for i in range(3):
            chan = arr[:,:,i].flatten()
            d = np.zeros_like(chan)
            d[0] = chan[0]
            for j in range(1, len(chan)): d[j] = chan[j] - chan[j-1]
            diff_channels.append(d)
            visuals.append(normalize_diff_image(d.reshape(rows, cols)))
        all_diffs = np.concatenate(diff_channels)
        write_image_to_file(np_to_PIL(np.stack(visuals, axis=2)), str(self.output_dir / f"{self.filename}_rgb_diff_visual.bmp"))
        codes, logs = self.encode("".join(chr(int(d) + 255) for d in all_diffs), is_diff=True)
        self.save_log_csv(logs, "rgb_diff_encode_log")
        with open(self.output_dir / f"{self.filename}_rgb_diff.bin", 'wb') as f:
            f.write(rows.to_bytes(4, 'big') + cols.to_bytes(4, 'big') + self.pack_binary(codes))

    def decompress_RGB_Diff(self):
        with open(self.output_dir / f"{self.filename}_rgb_diff.bin", 'rb') as f:
            data = f.read()
            rows, cols = int.from_bytes(data[:4], 'big'), int.from_bytes(data[4:8], 'big')
            codes = self.unpack_binary(data[8:], is_diff=True)
        decoded_str, logs = self.decode(codes, is_diff=True)
        self.save_log_csv(logs, "rgb_diff_decode_log")
        raw_diff = np.array([ord(x) - 255 for x in decoded_str])
        plane_size = rows * cols
        channels = []
        for i in range(3):
            segment = raw_diff[i*plane_size : (i+1)*plane_size]
            plane = np.zeros_like(segment)
            if len(segment) > 0:
                plane[0] = segment[0]
                for j in range(1, len(segment)): plane[j] = plane[j-1] + segment[j]
            channels.append(plane.reshape(rows, cols))
        result = np.stack(channels, axis=2).clip(0, 255).astype(np.uint8)
        write_image_to_file(np_to_PIL(result), str(self.output_dir / f"{self.filename}_rgb_diff_restored.bmp"))