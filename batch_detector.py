import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
from datetime import datetime
from model_engine import predict, generate_gradcam, save_scan, get_tumor_info
from db_config import get_connection

BG_DARK    = "#0A0E1A"
BG_PANEL   = "#0F1729"
BG_CARD    = "#162040"
ACCENT     = "#00C8FF"
ACCENT2    = "#7B2FFF"
TEXT_WHITE = "#FFFFFF"
TEXT_GRAY  = "#8BA3C7"
SUCCESS    = "#00C896"
DANGER     = "#FF4C4C"
WARNING    = "#FFB84C"


class BatchDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Detection — Batch Analyser")
        self.root.geometry("1100x700")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)
        self.center(1100, 700)
        self.image_paths = []
        self.results     = []
        self.build_ui()

    def center(self, w, h):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=BG_PANEL, height=62)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="📦  BATCH MRI ANALYSER",
                 font=("Segoe UI", 13, "bold"),
                 bg=BG_PANEL, fg=TEXT_WHITE).pack(side="left", padx=22, pady=15)
        tk.Button(hdr, text="⬅  Single Scan",
                  font=("Segoe UI", 9), bg=BG_CARD,
                  fg=TEXT_GRAY, relief="flat", cursor="hand2",
                  command=self.go_single).pack(side="right", padx=12, pady=14)

        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=15, pady=12)

        # Left — file list
        left = tk.Frame(body, bg=BG_DARK, width=380)
        left.pack(side="left", fill="both", padx=(0, 12))
        left.pack_propagate(False)

        tk.Label(left, text="MRI FILES",
                 font=("Segoe UI", 9, "bold"),
                 bg=BG_DARK, fg=TEXT_GRAY).pack(anchor="w", pady=(0, 5))

        self.file_listbox = tk.Listbox(left, bg=BG_CARD, fg=TEXT_WHITE,
                                        selectbackground=ACCENT,
                                        font=("Segoe UI", 9),
                                        relief="flat",
                                        highlightthickness=0)
        self.file_listbox.pack(fill="both", expand=True)

        btn_row = tk.Frame(left, bg=BG_DARK)
        btn_row.pack(fill="x", pady=(8, 0))
        tk.Button(btn_row, text="➕  Add Images",
                  font=("Segoe UI", 9),
                  bg=BG_CARD, fg=ACCENT, relief="flat",
                  cursor="hand2",
                  command=self.add_images).pack(side="left", ipady=8,
                  fill="x", expand=True, padx=(0, 6))
        tk.Button(btn_row, text="🗑  Clear All",
                  font=("Segoe UI", 9),
                  bg=BG_CARD, fg=DANGER, relief="flat",
                  cursor="hand2",
                  command=self.clear_files).pack(side="left", ipady=8,
                  fill="x", expand=True)

        # Progress
        self.progress_var  = tk.DoubleVar()
        self.progress_label = tk.Label(left, text="",
                                        font=("Segoe UI", 8),
                                        bg=BG_DARK, fg=TEXT_GRAY)
        self.progress_label.pack(pady=(6, 2))
        self.progress_bar = ttk.Progressbar(left, variable=self.progress_var,
                                             maximum=100, length=360)
        self.progress_bar.pack(fill="x", pady=(0, 6))

        tk.Button(left, text="🔬  ANALYSE ALL",
                  font=("Segoe UI", 11, "bold"),
                  bg=ACCENT, fg=BG_DARK, relief="flat",
                  cursor="hand2", activebackground="#00A8D8",
                  command=self.analyse_batch).pack(fill="x", ipady=12)

        # Right — results table
        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True)

        tk.Label(right, text="BATCH RESULTS",
                 font=("Segoe UI", 9, "bold"),
                 bg=BG_DARK, fg=TEXT_GRAY).pack(anchor="w", pady=(0, 5))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Batch.Treeview",
                         background=BG_CARD, foreground=TEXT_WHITE,
                         fieldbackground=BG_CARD, rowheight=30,
                         font=("Segoe UI", 9))
        style.configure("Batch.Treeview.Heading",
                         background=BG_PANEL, foreground=ACCENT,
                         font=("Segoe UI", 9, "bold"))
        style.map("Batch.Treeview",
                  background=[("selected", ACCENT2)])

        cols = ("File", "Prediction", "Confidence", "Severity", "Status")
        self.results_tree = ttk.Treeview(right, columns=cols,
                                          show="headings",
                                          style="Batch.Treeview")
        for col, w in zip(cols, [200, 150, 100, 90, 80]):
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=w)

        sb = ttk.Scrollbar(right, orient="vertical",
                            command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=sb.set)
        self.results_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Summary row
        self.summary_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.summary_var,
                 font=("Segoe UI", 9, "bold"),
                 bg=BG_DARK, fg=ACCENT).pack(anchor="w", pady=6)

    def add_images(self):
        paths = filedialog.askopenfilenames(
            title="Select MRI Images",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        for p in paths:
            if p not in self.image_paths:
                self.image_paths.append(p)
                self.file_listbox.insert("end", os.path.basename(p))

    def clear_files(self):
        self.image_paths = []
        self.file_listbox.delete(0, "end")
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.summary_var.set("")
        self.progress_var.set(0)
        self.progress_label.config(text="")

    def analyse_batch(self):
        if not self.image_paths:
            messagebox.showwarning("No Files", "Please add MRI images first.")
            return
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.results = []
        threading.Thread(target=self._run_batch, daemon=True).start()

    def _run_batch(self):
        total = len(self.image_paths)
        for i, path in enumerate(self.image_paths):
            try:
                preds     = predict(path)
                top       = preds[0]
                info      = get_tumor_info(top[0])
                severity  = info["severity"] if info else "Unknown"
                heatmap   = generate_gradcam(path)
                save_scan("Batch Patient", 0, "Unknown",
                           path, heatmap, preds,
                           top[0], severity,
                           info["recommendation"] if info else "")

                tag = "critical" if severity in ("Critical","High") else "normal"
                self.root.after(0, lambda f=os.path.basename(path),
                                       p=top[0], c=top[1], s=severity, t=tag:
                    self._add_result_row(f, p, c, s, t))

                pct = ((i + 1) / total) * 100
                self.root.after(0, lambda p=pct, n=i+1, tot=total:
                    self._update_progress(p, n, tot))

            except Exception as e:
                self.root.after(0, lambda f=os.path.basename(path), err=str(e):
                    self._add_result_row(f, "ERROR", 0, "—", "error"))

        self.root.after(0, self._batch_complete)

    def _add_result_row(self, filename, pred, conf, severity, tag):
        self.results_tree.insert("", "end",
            values=(filename, pred, f"{conf:.1f}%", severity, "✅ Done"),
            tags=(tag,))
        self.results_tree.tag_configure("critical", foreground=DANGER)
        self.results_tree.tag_configure("normal",   foreground=SUCCESS)
        self.results_tree.tag_configure("error",    foreground=WARNING)

    def _update_progress(self, pct, n, total):
        self.progress_var.set(pct)
        self.progress_label.config(text=f"Processing {n} of {total}...")

    def _batch_complete(self):
        self.progress_label.config(text="✅ Batch complete!")
        messagebox.showinfo("Done", f"Batch analysis complete!\n{len(self.image_paths)} images processed.")

    def go_single(self):
        self.root.destroy()
        import detector
        detector.launch()


def launch():
    root = tk.Tk()
    BatchDetector(root)
    root.mainloop()

if __name__ == "__main__":
    launch()