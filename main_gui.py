import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from LZW import LZWCoding
from basic_image_operations import read_image_from_file, PIL_to_np

class LZWManagerApp:
    def __init__(self, root_win):
        self.window = root_win
        self.window.title("LZW Compression Suite")
        self.window.geometry("640x600") # Boyutu metrikler için biraz büyüttük
        self.file_path = tk.StringVar()
        self.mode_var = tk.StringVar(value="Grayscale")
        self.render_ui()

    def render_ui(self):
        tk.Label(self.window, text="Lossless LZW Processor", font=("Segoe UI", 16, "bold")).pack(pady=20)
        bar = tk.Frame(self.window)
        bar.pack(pady=10, padx=30, fill="x")
        tk.Entry(bar, textvariable=self.file_path, state="readonly").pack(side="left", expand=True, fill="x", padx=5)
        tk.Button(bar, text="Open File", command=self.pick_file).pack(side="right")

        self.pnl = tk.LabelFrame(self.window, text="Algorithm Settings", padx=15, pady=15)
        self.pnl.pack(pady=10, padx=30, fill="x")
        opts = [("Grayscale", "Grayscale"), ("Gray Diff", "Grayscale Diff"), ("RGB", "RGB"), ("RGB Diff", "RGB Diff")]
        for label, val in opts:
            tk.Radiobutton(self.pnl, text=label, variable=self.mode_var, value=val).pack(side="left", expand=True)

        btns = tk.Frame(self.window)
        btns.pack(pady=20)
        tk.Button(btns, text="Compress", bg="#2c3e50", fg="white", font=("Arial", 9, "bold"), width=14, height=2, command=self.run_compress).pack(side="left", padx=10)
        tk.Button(btns, text="Decompress", bg="#2980b9", fg="white", font=("Arial", 9, "bold"), width=14, height=2, command=self.run_decompress).pack(side="left", padx=10)

        self.log_box = tk.Text(self.window, height=12, bg="#fdfdfd", state="disabled", font=("Courier", 10))
        self.log_box.pack(pady=10, padx=30, fill="x")

    def pick_file(self):
        path = filedialog.askopenfilename(filetypes=[("All Compatible", "*.txt *.bmp *.bin")])
        if path:
            self.file_path.set(path)
            self.pnl.config(text="Mode: " + ("Text Processor" if path.endswith(".txt") else "Image Processor"))

    def write_log(self, msg):
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"> {msg}\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def show_results(self, original_path, restored_path):
        """Displays original and restored images side by side."""
        try:
            orig = Image.open(original_path)
            rest = Image.open(restored_path)
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(orig, cmap='gray' if 'gray' in str(restored_path).lower() else None)
            plt.title(f"Original\n({os.path.basename(original_path)})")
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(rest, cmap='gray' if 'gray' in str(restored_path).lower() else None)
            plt.title(f"Restored (LZW)\n({os.path.basename(restored_path)})")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            self.write_log(f"Display Error: {e}")

    def run_compress(self):
        p = self.file_path.get()
        if not p: return

        # --- Ekleme 1: Hatalı Dosya Kontrolü ---
        if not p.lower().endswith(('.bmp', '.txt')):
            messagebox.showerror("Error", "Unsupported file! Please use .bmp or .txt")
            return

        fname = os.path.basename(p).split('.')[0]
        lzw = LZWCoding(fname)
        try:
            if p.lower().endswith(".txt"):
                lzw.compress_text_file()
                res_p = lzw.output_dir / f"{fname}_text_compressed.bin"
            else:
                m = self.mode_var.get()
                if m == "Grayscale": lzw.compress_Grayscale()
                elif m == "Grayscale Diff": lzw.compress_Grayscale_Diff()
                elif m == "RGB": lzw.compress_RGB()
                elif m == "RGB Diff": lzw.compress_RGB_Diff()

                suffix = "_gray.bin" if m == "Grayscale" else \
                    "_gray_diff.bin" if m == "Grayscale Diff" else \
                        "_rgb.bin" if m == "RGB" else "_rgb_diff.bin"
                res_p = lzw.output_dir / (fname + suffix)

            # --- Ekleme 2: CR, CF, SS Metrikleri ---
            s_old = os.path.getsize(p)
            s_new = os.path.getsize(res_p)
            cr, cf, ss = lzw.calculate_metrics(s_old, s_new)

            self.write_log(f"SUCCESS: {fname}")
            self.write_log(f"  - CR (Ratio): {cr}")
            self.write_log(f"  - CF (Factor): %{cf}")
            self.write_log(f"  - SS (Saving): %{ss}")

            messagebox.showinfo("Done", f"Compression complete!\nSaved: {res_p.name}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_decompress(self):
        p = self.file_path.get()
        if not p.endswith(".bin"):
            messagebox.showwarning("Warning", "Select a .bin file.")
            return

        raw_name = os.path.basename(p)
        tags = ["_text_compressed.bin", "_gray_diff.bin", "_gray.bin", "_rgb_diff.bin", "_rgb.bin"]
        clean_name = raw_name
        for t in tags:
            if clean_name.endswith(t):
                clean_name = clean_name[:-len(t)]
                break

        lzw = LZWCoding(clean_name)
        try:
            restored_file = None
            if "text" in p:
                lzw.decompress_text_file()
                restored_file = lzw.output_dir / f"{clean_name}_text_restored.txt"
                os.system(f"open '{restored_file}'")
            else:
                if "gray_diff" in p:
                    lzw.decompress_Grayscale_Diff()
                    restored_file = lzw.output_dir / f"{clean_name}_gray_diff_restored.bmp"
                elif "gray" in p:
                    lzw.decompress_Grayscale()
                    restored_file = lzw.output_dir / f"{clean_name}_gray_restored.bmp"
                elif "rgb_diff" in p:
                    lzw.decompress_RGB_Diff()
                    restored_file = lzw.output_dir / f"{clean_name}_rgb_diff_restored.bmp"
                elif "rgb" in p:
                    lzw.decompress_RGB()
                    restored_file = lzw.output_dir / f"{clean_name}_rgb_restored.bmp"

                original_file = lzw.input_dir / f"{clean_name}.bmp"
                if restored_file and restored_file.exists():
                    self.show_results(original_file, restored_file)

            self.write_log(f"RESTORED: {raw_name}")
            messagebox.showinfo("Success", "Decompression complete.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    LZWManagerApp(root)
    root.mainloop()