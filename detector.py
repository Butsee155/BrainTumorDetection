import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
from model_engine import (predict, generate_gradcam,
                           highlight_tumor_region,
                           get_tumor_info, save_scan)
from datetime import datetime

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
CRITICAL   = "#FF2020"

SEVERITY_COLORS = {
    "Low":      "#00C896",
    "Medium":   "#FFB84C",
    "High":     "#FF6B35",
    "Critical": "#FF2020",
}


class Detector:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Detection — MRI Analyser")
        self.root.geometry("1200x740")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)
        self.center(1200, 740)
        self.current_image_path = None
        self.heatmap_path       = None
        self.annotated_path     = None
        self.predictions        = []
        self._clock_running     = True   # ← flag to stop clock safely
        self.build_ui()

    def center(self, w, h):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    # ── Build UI ──────────────────────────────────────────────────────────────
    def build_ui(self):

        # Header
        hdr = tk.Frame(self.root, bg=BG_PANEL, height=62)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="🧠  BRAIN TUMOR DETECTION — MRI ANALYSER",
                 font=("Segoe UI", 13, "bold"),
                 bg=BG_PANEL, fg=TEXT_WHITE).pack(side="left", padx=22, pady=15)

        self.clock_var = tk.StringVar()
        tk.Label(hdr, textvariable=self.clock_var,
                 font=("Segoe UI", 9), bg=BG_PANEL,
                 fg=ACCENT).pack(side="right", padx=22)

        btn_frame = tk.Frame(hdr, bg=BG_PANEL)
        btn_frame.pack(side="right", padx=8)
        tk.Button(btn_frame, text="📦  Batch",
                  font=("Segoe UI", 9), bg=ACCENT2,
                  fg=TEXT_WHITE, relief="flat", cursor="hand2",
                  command=self.open_batch).pack(side="left", padx=4, pady=14)
        tk.Button(btn_frame, text="⬅  Back",
                  font=("Segoe UI", 9), bg=BG_CARD,
                  fg=TEXT_GRAY, relief="flat", cursor="hand2",
                  command=self.go_back).pack(side="left", padx=4, pady=14)

        self.update_clock()

        # Body
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=15, pady=12)

        # ── Left panel ────────────────────────────────────────────────────────
        left = tk.Frame(body, bg=BG_DARK, width=560)
        left.pack(side="left", fill="both", padx=(0, 12))
        left.pack_propagate(False)

        # Patient info card
        info = tk.Frame(left, bg=BG_CARD)
        info.pack(fill="x", pady=(0, 10))
        tk.Label(info, text="PATIENT INFORMATION",
                 font=("Segoe UI", 8, "bold"),
                 bg=BG_CARD, fg=TEXT_GRAY).pack(anchor="w", padx=14, pady=(10, 5))
        tk.Frame(info, bg=ACCENT, height=1).pack(fill="x", padx=14)

        fields_row = tk.Frame(info, bg=BG_CARD)
        fields_row.pack(fill="x", padx=14, pady=8)

        self.info_vars = {}

        # ── Patient Name field ────────────────────────────────────────────────
        col_name = tk.Frame(fields_row, bg=BG_CARD)
        col_name.pack(side="left", padx=(0, 15))
        tk.Label(col_name, text="Patient Name", font=("Segoe UI", 8),
                 bg=BG_CARD, fg=TEXT_GRAY).pack(anchor="w")
        name_var = tk.StringVar()
        self.info_vars["name"] = name_var
        name_entry = tk.Entry(col_name, textvariable=name_var, width=22,
                              bg=BG_PANEL, fg=TEXT_WHITE,
                              insertbackground=TEXT_WHITE,
                              relief="flat", font=("Segoe UI", 10),
                              highlightbackground=ACCENT,
                              highlightthickness=1)
        name_entry.pack(ipady=6, padx=2)
        name_entry.insert(0, "e.g. John Silva")
        name_entry.bind("<FocusIn>",
                        lambda ev: name_entry.delete(0, "end")
                        if name_entry.get() == "e.g. John Silva" else None)
        name_entry.bind("<FocusOut>",
                        lambda ev: name_entry.insert(0, "e.g. John Silva")
                        if not name_entry.get() else None)

        # ── Age field ─────────────────────────────────────────────────────────
        col_age = tk.Frame(fields_row, bg=BG_CARD)
        col_age.pack(side="left", padx=(0, 15))
        tk.Label(col_age, text="Age", font=("Segoe UI", 8),
                 bg=BG_CARD, fg=TEXT_GRAY).pack(anchor="w")
        age_var = tk.StringVar()
        self.info_vars["age"] = age_var
        age_entry = tk.Entry(col_age, textvariable=age_var, width=8,
                             bg=BG_PANEL, fg=TEXT_WHITE,
                             insertbackground=TEXT_WHITE,
                             relief="flat", font=("Segoe UI", 10),
                             highlightbackground=ACCENT,
                             highlightthickness=1)
        age_entry.pack(ipady=6, padx=2)
        age_entry.insert(0, "e.g. 35")
        age_entry.bind("<FocusIn>",
                       lambda ev: age_entry.delete(0, "end")
                       if age_entry.get() == "e.g. 35" else None)
        age_entry.bind("<FocusOut>",
                       lambda ev: age_entry.insert(0, "e.g. 35")
                       if not age_entry.get() else None)

        # ── Gender ────────────────────────────────────────────────────────────
        col_gender = tk.Frame(fields_row, bg=BG_CARD)
        col_gender.pack(side="left")
        tk.Label(col_gender, text="Gender", font=("Segoe UI", 8),
                 bg=BG_CARD, fg=TEXT_GRAY).pack(anchor="w")
        self.gender_var = tk.StringVar(value="Male")
        g_row = tk.Frame(col_gender, bg=BG_CARD)
        g_row.pack()
        for g in ["Male", "Female", "Other"]:
            tk.Radiobutton(g_row, text=g, variable=self.gender_var, value=g,
                           bg=BG_CARD, fg=TEXT_WHITE,
                           selectcolor=BG_DARK,
                           activebackground=BG_CARD,
                           font=("Segoe UI", 9)).pack(side="left", padx=4)

        # ── Image display tabs ────────────────────────────────────────────────
        tab_frame = tk.Frame(left, bg=BG_DARK)
        tab_frame.pack(fill="both", expand=True)

        self.view_var = tk.StringVar(value="original")
        tab_btn_row = tk.Frame(tab_frame, bg=BG_DARK)
        tab_btn_row.pack(fill="x")
        for text, val in [("Original",  "original"),
                           ("Grad-CAM",  "gradcam"),
                           ("Annotated", "annotated")]:
            tk.Radiobutton(tab_btn_row, text=text,
                           variable=self.view_var, value=val,
                           bg=BG_DARK, fg=TEXT_GRAY,
                           selectcolor=BG_DARK,
                           activebackground=BG_DARK,
                           font=("Segoe UI", 9),
                           command=self.switch_view).pack(side="left", padx=8, pady=4)

        self.img_label = tk.Label(tab_frame, bg=BG_CARD,
                                   text="Upload MRI image\nto begin analysis",
                                   font=("Segoe UI", 11),
                                   fg=TEXT_GRAY, width=50, height=20)
        self.img_label.pack(fill="both", expand=True, pady=(4, 0))

        # ── Upload + Analyse buttons ──────────────────────────────────────────
        btn_row = tk.Frame(left, bg=BG_DARK)
        btn_row.pack(fill="x", pady=(10, 0))
        tk.Button(btn_row, text="📂  Upload MRI Image",
                  font=("Segoe UI", 10, "bold"),
                  bg=BG_CARD, fg=ACCENT, relief="flat",
                  cursor="hand2", activebackground=BG_PANEL,
                  command=self.upload_image).pack(side="left", ipady=10,
                  padx=(0, 8), fill="x", expand=True)
        tk.Button(btn_row, text="🔬  Analyse",
                  font=("Segoe UI", 10, "bold"),
                  bg=ACCENT, fg=BG_DARK, relief="flat",
                  cursor="hand2", activebackground="#00A8D8",
                  command=self.analyse).pack(side="left", ipady=10,
                  fill="x", expand=True)

        # ── Right results panel ───────────────────────────────────────────────
        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True)
        self.build_results_panel(right)

    # ── Results Panel ─────────────────────────────────────────────────────────
    def build_results_panel(self, parent):
        hdr = tk.Frame(parent, bg=BG_PANEL, height=46)
        hdr.pack(fill="x", pady=(0, 10))
        hdr.pack_propagate(False)
        tk.Label(hdr, text="📊  ANALYSIS RESULTS",
                 font=("Segoe UI", 11, "bold"),
                 bg=BG_PANEL, fg=TEXT_WHITE).pack(side="left", padx=14, pady=10)
        tk.Button(hdr, text="🔄 Clear",
                  font=("Segoe UI", 8), bg=BG_CARD,
                  fg=TEXT_GRAY, relief="flat", cursor="hand2",
                  command=self.clear_results).pack(side="right", padx=14)

        rc = tk.Canvas(parent, bg=BG_DARK, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=rc.yview)
        self.results_frame = tk.Frame(rc, bg=BG_DARK)
        self.results_frame.bind("<Configure>",
            lambda ev: rc.configure(scrollregion=rc.bbox("all")))
        rc.create_window((0, 0), window=self.results_frame, anchor="nw")
        rc.configure(yscrollcommand=sb.set)
        rc.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.status_label = tk.Label(self.results_frame,
            text="Upload an MRI image and click Analyse",
            font=("Segoe UI", 11), bg=BG_DARK,
            fg=TEXT_GRAY, justify="center")
        self.status_label.pack(pady=60)

    # ── Upload Image ──────────────────────────────────────────────────────────
    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if not path:
            return
        self.current_image_path = path
        self.heatmap_path       = None
        self.annotated_path     = None
        self.show_image(path)
        self.status_label.config(
            text=f"Image loaded:\n{os.path.basename(path)}\n\nClick Analyse to begin")

    def show_image(self, path):
        try:
            img   = Image.open(path).convert("RGB")
            img   = img.resize((520, 380), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(img)
            self.img_label.imgtk = imgtk
            self.img_label.config(image=imgtk, text="")
        except Exception as img_err:
            print(f"[ERROR] Show image: {img_err}")

    def switch_view(self):
        view = self.view_var.get()
        if view == "original" and self.current_image_path:
            self.show_image(self.current_image_path)
        elif view == "gradcam" and self.heatmap_path:
            self.show_image(self.heatmap_path)
        elif view == "annotated" and self.annotated_path:
            self.show_image(self.annotated_path)
        else:
            self.img_label.config(text="Generate analysis first", image="")

    # ── Analyse ───────────────────────────────────────────────────────────────
    def analyse(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image",
                                    "Please upload an MRI image first.")
            return

        name   = self.info_vars["name"].get().strip()
        age    = self.info_vars["age"].get().strip()
        gender = self.gender_var.get()

        if name in ["", "e.g. John Silva"]:
            messagebox.showwarning("Missing", "Please enter patient name.")
            return
        if age in ["", "e.g. 35"]:
            messagebox.showwarning("Missing", "Please enter patient age.")
            return

        # Show loading
        for w in self.results_frame.winfo_children():
            w.destroy()
        self.status_label = tk.Label(self.results_frame,
            text="⏳  Analysing MRI scan...\nPlease wait...",
            font=("Segoe UI", 12), bg=BG_DARK,
            fg=ACCENT, justify="center")
        self.status_label.pack(pady=60)
        self.root.update()

        # Run analysis in background thread
        threading.Thread(
            target=self._run_analysis,
            args=(name, age, gender),
            daemon=True).start()

    def _run_analysis(self, name, age, gender):
        try:
            # Step 1: Predict
            self.predictions = predict(self.current_image_path)
            top_class        = self.predictions[0][0]
            class_idx        = ["Glioma", "Meningioma",
                                 "No Tumor", "Pituitary"].index(top_class)

            # Step 2: Grad-CAM heatmap
            self.heatmap_path = generate_gradcam(
                self.current_image_path, class_idx)

            # Step 3: Annotated with bounding box
            self.annotated_path = highlight_tumor_region(
                self.current_image_path, self.heatmap_path)

            # Step 4: Get tumor details from DB
            details        = get_tumor_info(top_class)
            severity       = details["severity"]       if details else "Unknown"
            recommendation = details["recommendation"] if details else "Consult a doctor."

            # Step 5: Save to database
            save_scan(name, age, gender,
                       self.current_image_path,
                       self.heatmap_path,
                       self.predictions,
                       top_class, severity, recommendation)

            # Step 6: Update UI on main thread
            self.root.after(0, lambda: self.show_results(
                self.predictions, details, name))

        except Exception as analysis_err:
            # ── FIX: renamed 'e' to 'analysis_err' to avoid lambda conflict ──
            err_msg = str(analysis_err)
            self.root.after(0, lambda: messagebox.showerror(
                "Analysis Error",
                f"Analysis failed:\n{err_msg}"))

    # ── Show Results ──────────────────────────────────────────────────────────
    def show_results(self, predictions, details, name):
        for w in self.results_frame.winfo_children():
            w.destroy()

        pad = {"padx": 12, "pady": 6}

        # Patient summary card
        p_card = tk.Frame(self.results_frame, bg=BG_CARD)
        p_card.pack(fill="x", **pad)
        tk.Label(p_card,
                 text=f"Patient: {name}  |  "
                      f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
                 font=("Segoe UI", 8), bg=BG_CARD,
                 fg=TEXT_GRAY).pack(anchor="w", padx=12, pady=8)
        tk.Label(p_card,
                 text=f"Image: {os.path.basename(self.current_image_path)}",
                 font=("Segoe UI", 8), bg=BG_CARD,
                 fg=TEXT_GRAY).pack(anchor="w", padx=12, pady=(0, 8))

        # Top 3 predictions
        tk.Label(self.results_frame, text="TOP PREDICTIONS",
                 font=("Segoe UI", 8, "bold"),
                 bg=BG_DARK, fg=TEXT_GRAY).pack(anchor="w", padx=12, pady=(8, 4))

        colors = [ACCENT, ACCENT2, "#4A9FFF"]
        for i, (cond, conf) in enumerate(predictions):
            card = tk.Frame(self.results_frame, bg=BG_CARD)
            card.pack(fill="x", **pad)

            stripe = tk.Frame(card, bg=colors[i], width=5)
            stripe.pack(side="left", fill="y")

            inner = tk.Frame(card, bg=BG_CARD)
            inner.pack(side="left", fill="both",
                       expand=True, padx=12, pady=10)

            rank = ["🥇 Most Likely", "🥈 Possible", "🥉 Less Likely"][i]
            tk.Label(inner, text=rank,
                     font=("Segoe UI", 7, "bold"),
                     bg=BG_CARD, fg=colors[i]).pack(anchor="w")
            tk.Label(inner, text=cond,
                     font=("Segoe UI", 12, "bold"),
                     bg=BG_CARD, fg=TEXT_WHITE).pack(anchor="w")

            # Confidence bar
            bar_row = tk.Frame(inner, bg=BG_CARD)
            bar_row.pack(fill="x", pady=(3, 0))
            tk.Label(bar_row, text=f"{conf}%",
                     font=("Segoe UI", 8, "bold"),
                     bg=BG_CARD, fg=colors[i]).pack(side="right")
            bar_bg = tk.Frame(bar_row, bg=BG_PANEL, height=8)
            bar_bg.pack(side="left", fill="x",
                        expand=True, padx=(0, 8))
            tk.Frame(bar_bg, bg=colors[i],
                     height=8,
                     width=int(conf * 2.5)).pack(side="left")

        # Condition details
        if details:
            sev_color = SEVERITY_COLORS.get(details["severity"], ACCENT)

            tk.Label(self.results_frame, text="DIAGNOSIS DETAILS",
                     font=("Segoe UI", 8, "bold"),
                     bg=BG_DARK, fg=TEXT_GRAY).pack(
                anchor="w", padx=12, pady=(10, 4))

            det = tk.Frame(self.results_frame, bg=BG_CARD)
            det.pack(fill="x", **pad)

            # Severity badge
            sev_row = tk.Frame(det, bg=BG_CARD)
            sev_row.pack(fill="x", padx=12, pady=(10, 4))
            tk.Label(sev_row, text="Severity:",
                     font=("Segoe UI", 9),
                     bg=BG_CARD, fg=TEXT_GRAY).pack(side="left")
            tk.Label(sev_row,
                     text=f"  {details['severity'].upper()}  ",
                     font=("Segoe UI", 9, "bold"),
                     bg=sev_color, fg=TEXT_WHITE).pack(side="left", padx=8)

            # Urgency
            tk.Label(det, text=details["urgency"],
                     font=("Segoe UI", 9, "bold"),
                     bg=BG_CARD, fg=sev_color).pack(
                anchor="w", padx=12, pady=2)

            # Description
            tk.Label(det, text=details["description"],
                     font=("Segoe UI", 9), bg=BG_CARD,
                     fg=TEXT_GRAY, wraplength=340,
                     justify="left").pack(anchor="w", padx=12, pady=4)

            # Recommendation
            rec = tk.Frame(det, bg="#0A2E1A")
            rec.pack(fill="x", padx=12, pady=6)
            tk.Label(rec, text="✅  Recommendation:",
                     font=("Segoe UI", 8, "bold"),
                     bg="#0A2E1A", fg=SUCCESS).pack(
                anchor="w", padx=8, pady=(6, 2))
            tk.Label(rec, text=details["recommendation"],
                     font=("Segoe UI", 9), bg="#0A2E1A",
                     fg="#90EE90", wraplength=340,
                     justify="left").pack(
                anchor="w", padx=8, pady=(0, 6))

            # Specialist
            spec = tk.Frame(det, bg="#0A1E3A")
            spec.pack(fill="x", padx=12, pady=(0, 10))
            tk.Label(spec,
                     text=f"👨‍⚕️  Refer to: {details['specialist']}",
                     font=("Segoe UI", 9, "bold"),
                     bg="#0A1E3A", fg=ACCENT).pack(
                anchor="w", padx=8, pady=8)

        # View toggle hint
        tk.Label(self.results_frame,
                 text="💡 Use the tabs above to switch between\n"
                      "Original  /  Grad-CAM  /  Annotated views",
                 font=("Segoe UI", 8), bg=BG_DARK,
                 fg=TEXT_GRAY, justify="center").pack(pady=10)

        # Auto switch to Grad-CAM view
        self.view_var.set("gradcam")
        self.switch_view()

        # Disclaimer
        tk.Label(self.results_frame,
                 text="⚠  For research and educational purposes only.\n"
                      "Always consult a qualified medical professional.",
                 font=("Segoe UI", 8), bg=BG_DARK,
                 fg=DANGER, justify="center").pack(pady=10)

    # ── Clear Results ─────────────────────────────────────────────────────────
    def clear_results(self):
        for w in self.results_frame.winfo_children():
            w.destroy()
        self.status_label = tk.Label(self.results_frame,
            text="Upload an MRI image and click Analyse",
            font=("Segoe UI", 11), bg=BG_DARK,
            fg=TEXT_GRAY, justify="center")
        self.status_label.pack(pady=60)
        self.current_image_path = None
        self.heatmap_path       = None
        self.annotated_path     = None
        self.img_label.config(image="",
            text="Upload MRI image\nto begin analysis")

    # ── Clock (FIXED — stops safely when window closes) ───────────────────────
    def update_clock(self):
        try:
            self.clock_var.set(
                datetime.now().strftime("%A  %d %b %Y  |  %H:%M:%S"))
            self.root.after(1000, self.update_clock)
        except tk.TclError:
            pass  # window destroyed — stop cleanly

    # ── Navigation ────────────────────────────────────────────────────────────
    def open_batch(self):
        self._clock_running = False
        self.root.destroy()
        import batch_detector
        batch_detector.launch()

    def go_back(self):
        self._clock_running = False
        self.root.destroy()
        import main_app
        main_app.launch()


def launch():
    root = tk.Tk()
    Detector(root)
    root.mainloop()


if __name__ == "__main__":
    launch()