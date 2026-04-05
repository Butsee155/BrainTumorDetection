import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import openpyxl, csv, os
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


class Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Detection — Dashboard")
        self.root.geometry("1100x700")
        self.root.configure(bg=BG_DARK)
        self.center(1100, 700)
        self.build_ui()
        self.load_stats()

    def center(self, w, h):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def build_ui(self):
        # Sidebar
        sidebar = tk.Frame(self.root, bg=BG_PANEL, width=215)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="🧠", font=("Segoe UI", 32),
                 bg=BG_PANEL, fg=ACCENT).pack(pady=(25, 4))
        tk.Label(sidebar, text="BRAIN TUMOR",
                 font=("Segoe UI", 11, "bold"),
                 bg=BG_PANEL, fg=TEXT_WHITE).pack()
        tk.Label(sidebar, text="Admin Dashboard",
                 font=("Segoe UI", 9),
                 bg=BG_PANEL, fg=TEXT_GRAY).pack(pady=(0, 20))
        tk.Frame(sidebar, bg=ACCENT, height=1).pack(fill="x", padx=20)

        self.nav_btns = {}
        for label, key in [
            ("📊  Overview",    "overview"),
            ("📋  Scan History","history"),
            ("📤  Export",      "export"),
        ]:
            btn = tk.Button(sidebar, text=label,
                            font=("Segoe UI", 10),
                            bg=BG_PANEL, fg=TEXT_GRAY,
                            relief="flat", cursor="hand2",
                            anchor="w", padx=20,
                            activebackground=BG_CARD,
                            activeforeground=TEXT_WHITE,
                            command=lambda k=key: self.show_page(k))
            btn.pack(fill="x", ipady=10, pady=1)
            self.nav_btns[key] = btn

        tk.Frame(sidebar, bg=BG_CARD, height=1).pack(fill="x", padx=20, pady=15)
        tk.Button(sidebar, text="🔬  Open Analyser",
                  font=("Segoe UI", 10, "bold"),
                  bg=ACCENT, fg=BG_DARK, relief="flat",
                  cursor="hand2", padx=20,
                  activebackground="#00A8D8",
                  command=self.open_detector).pack(fill="x", ipady=10)
        tk.Button(sidebar, text="🚪  Logout",
                  font=("Segoe UI", 10),
                  bg=BG_PANEL, fg=DANGER, relief="flat",
                  cursor="hand2", padx=20,
                  activebackground=BG_CARD,
                  command=self.logout).pack(fill="x", ipady=8, pady=5)

        self.content = tk.Frame(self.root, bg=BG_DARK)
        self.content.pack(side="left", fill="both", expand=True)

        self.pages = {}
        self.build_overview()
        self.build_history()
        self.build_export()
        self.show_page("overview")

    def show_page(self, key):
        for f in self.pages.values():
            f.pack_forget()
        for k, b in self.nav_btns.items():
            b.config(bg=BG_PANEL, fg=TEXT_GRAY)
        self.pages[key].pack(fill="both", expand=True)
        self.nav_btns[key].config(bg=BG_CARD, fg=TEXT_WHITE)
        if key == "history":  self.load_history()
        if key == "overview": self.load_stats()

    def build_overview(self):
        page = tk.Frame(self.content, bg=BG_DARK)
        self.pages["overview"] = page

        hdr = tk.Frame(page, bg=BG_PANEL, height=68)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="📊  System Overview",
                 font=("Segoe UI", 14, "bold"),
                 bg=BG_PANEL, fg=TEXT_WHITE).pack(side="left", padx=25, pady=20)
        tk.Button(hdr, text="🔄 Refresh",
                  font=("Segoe UI", 9),
                  bg=ACCENT, fg=BG_DARK, relief="flat",
                  cursor="hand2",
                  command=self.load_stats).pack(side="right", padx=20, pady=18)

        cards = tk.Frame(page, bg=BG_DARK)
        cards.pack(fill="x", padx=22, pady=18)

        self.stat_vars = {}
        for i, (icon, label, key, color) in enumerate([
            ("🔬", "Total Scans",    "total",      ACCENT),
            ("🔴", "Glioma Cases",   "glioma",     DANGER),
            ("🟠", "Meningioma",     "meningioma", WARNING),
            ("🟡", "Pituitary",      "pituitary",  "#FFD700"),
            ("🟢", "No Tumor",       "notumor",    SUCCESS),
        ]):
            card = tk.Frame(cards, bg=BG_CARD, height=108)
            card.grid(row=0, column=i, padx=6, sticky="nsew")
            card.pack_propagate(False)
            cards.columnconfigure(i, weight=1)
            tk.Frame(card, bg=color, width=4).pack(side="left", fill="y")
            inner = tk.Frame(card, bg=BG_CARD)
            inner.pack(side="left", fill="both", expand=True, padx=12, pady=12)
            tk.Label(inner, text=icon, font=("Segoe UI", 20),
                     bg=BG_CARD, fg=color).pack(anchor="w")
            var = tk.StringVar(value="0")
            self.stat_vars[key] = var
            tk.Label(inner, textvariable=var,
                     font=("Segoe UI", 20, "bold"),
                     bg=BG_CARD, fg=TEXT_WHITE).pack(anchor="w")
            tk.Label(inner, text=label,
                     font=("Segoe UI", 7.5),
                     bg=BG_CARD, fg=TEXT_GRAY).pack(anchor="w")

        tk.Label(page, text="Recent Scans",
                 font=("Segoe UI", 11, "bold"),
                 bg=BG_DARK, fg=TEXT_WHITE).pack(anchor="w", padx=22, pady=(8,5))

        tbl = tk.Frame(page, bg=BG_DARK)
        tbl.pack(fill="both", expand=True, padx=22, pady=(0,20))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Brain.Treeview",
                         background=BG_CARD, foreground=TEXT_WHITE,
                         fieldbackground=BG_CARD, rowheight=30,
                         font=("Segoe UI", 9))
        style.configure("Brain.Treeview.Heading",
                         background=BG_PANEL, foreground=ACCENT,
                         font=("Segoe UI", 9, "bold"))
        style.map("Brain.Treeview",
                  background=[("selected", ACCENT2)])

        cols = ("Patient", "Age", "Diagnosis", "Confidence", "Severity", "Time")
        self.ov_tree = ttk.Treeview(tbl, columns=cols,
                                     show="headings",
                                     style="Brain.Treeview", height=10)
        for col, w in zip(cols, [150, 50, 150, 100, 90, 160]):
            self.ov_tree.heading(col, text=col)
            self.ov_tree.column(col, width=w)

        sb = ttk.Scrollbar(tbl, orient="vertical", command=self.ov_tree.yview)
        self.ov_tree.configure(yscrollcommand=sb.set)
        self.ov_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def load_stats(self):
        try:
            conn   = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM ScanHistory")
            self.stat_vars["total"].set(str(cursor.fetchone()[0]))
            for key, val in [("glioma",     "Glioma"),
                              ("meningioma", "Meningioma"),
                              ("pituitary",  "Pituitary"),
                              ("notumor",    "No Tumor")]:
                cursor.execute(
                    "SELECT COUNT(*) FROM ScanHistory WHERE Prediction1=?", val)
                self.stat_vars[key].set(str(cursor.fetchone()[0]))

            cursor.execute("""
                SELECT TOP 10 PatientName, PatientAge, Prediction1,
                       Confidence1, SeverityLevel, ScannedAt
                FROM ScanHistory ORDER BY ScannedAt DESC
            """)
            for item in self.ov_tree.get_children():
                self.ov_tree.delete(item)
            for row in cursor.fetchall():
                vals    = list(row)
                vals[3] = f"{vals[3]:.1f}%"
                sev     = row[4] or ""
                tag     = "critical" if sev in ("Critical","High") else "normal"
                self.ov_tree.insert("", "end", values=vals, tags=(tag,))
            self.ov_tree.tag_configure("critical", foreground=DANGER)
            self.ov_tree.tag_configure("normal",   foreground=SUCCESS)
            conn.close()
        except Exception as e:
            print(f"Stats error: {e}")

    def build_history(self):
        page = tk.Frame(self.content, bg=BG_DARK)
        self.pages["history"] = page

        hdr = tk.Frame(page, bg=BG_PANEL, height=68)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="📋  Scan History",
                 font=("Segoe UI", 14, "bold"),
                 bg=BG_PANEL, fg=TEXT_WHITE).pack(side="left", padx=25, pady=20)
        tk.Button(hdr, text="🔄 Refresh",
                  font=("Segoe UI", 9),
                  bg=ACCENT, fg=BG_DARK,
                  relief="flat", cursor="hand2",
                  command=self.load_history).pack(side="right", padx=25, pady=18)

        tbl = tk.Frame(page, bg=BG_DARK)
        tbl.pack(fill="both", expand=True, padx=22, pady=15)

        cols = ("ID", "Patient", "Age", "Gender",
                "Diagnosis", "Confidence", "Severity", "Time")
        self.hist_tree = ttk.Treeview(tbl, columns=cols,
                                       show="headings",
                                       style="Brain.Treeview")
        for col, w in zip(cols, [50, 150, 40, 70, 140, 90, 90, 150]):
            self.hist_tree.heading(col, text=col)
            self.hist_tree.column(col, width=w)

        sb = ttk.Scrollbar(tbl, orient="vertical", command=self.hist_tree.yview)
        self.hist_tree.configure(yscrollcommand=sb.set)
        self.hist_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def load_history(self):
        try:
            conn   = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT TOP 300 ScanID, PatientName, PatientAge, PatientGender,
                       Prediction1, Confidence1, SeverityLevel, ScannedAt
                FROM ScanHistory ORDER BY ScannedAt DESC
            """)
            for item in self.hist_tree.get_children():
                self.hist_tree.delete(item)
            for row in cursor.fetchall():
                vals    = list(row)
                vals[5] = f"{vals[5]:.1f}%" if vals[5] else "—"
                self.hist_tree.insert("", "end", values=vals)
            conn.close()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def build_export(self):
        page = tk.Frame(self.content, bg=BG_DARK)
        self.pages["export"] = page

        hdr = tk.Frame(page, bg=BG_PANEL, height=68)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="📤  Export Reports",
                 font=("Segoe UI", 14, "bold"),
                 bg=BG_PANEL, fg=TEXT_WHITE).pack(side="left", padx=25, pady=20)

        cards = tk.Frame(page, bg=BG_DARK)
        cards.pack(fill="both", expand=True, padx=40, pady=35)

        for i, (icon, title, cmd) in enumerate([
            ("📋", "All Scan History",    self.exp_all),
            ("📅", "Today's Scans",       self.exp_today),
            ("🔴", "Critical Cases",      self.exp_critical),
            ("📊", "Diagnosis Summary",   self.exp_summary),
        ]):
            card = tk.Frame(cards, bg=BG_CARD, height=165)
            card.grid(row=i//2, column=i%2, padx=14, pady=14, sticky="nsew")
            card.pack_propagate(False)
            cards.columnconfigure(i%2, weight=1)
            tk.Label(card, text=icon, font=("Segoe UI", 34),
                     bg=BG_CARD, fg=ACCENT).pack(pady=(18, 4))
            tk.Label(card, text=title,
                     font=("Segoe UI", 11, "bold"),
                     bg=BG_CARD, fg=TEXT_WHITE).pack()
            br = tk.Frame(card, bg=BG_CARD)
            br.pack(pady=10)
            tk.Button(br, text="Excel", font=("Segoe UI", 9),
                      bg=SUCCESS, fg=BG_DARK, relief="flat",
                      cursor="hand2", padx=10,
                      command=lambda c=cmd: c("xlsx")).pack(side="left", padx=3)
            tk.Button(br, text="CSV", font=("Segoe UI", 9),
                      bg=ACCENT, fg=BG_DARK, relief="flat",
                      cursor="hand2", padx=10,
                      command=lambda c=cmd: c("csv")).pack(side="left", padx=3)

    def _save(self, fmt, name):
        ext = ".xlsx" if fmt == "xlsx" else ".csv"
        return filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[("Excel","*.xlsx"),("CSV","*.csv")],
            initialfile=name)

    def _write(self, path, headers, rows, fmt):
        if not path: return
        if fmt == "xlsx":
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.append(headers)
            for r in rows: ws.append(list(r))
            wb.save(path)
        else:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(headers)
                w.writerows(rows)
        messagebox.showinfo("Exported", f"Saved:\n{path}")
        os.startfile(path)

    def _fetch(self, where=""):
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT ScanID, PatientName, PatientAge, PatientGender,
                   Prediction1, Confidence1, Prediction2, Confidence2,
                   Prediction3, Confidence3, FinalDiagnosis,
                   SeverityLevel, Recommendation, ScannedAt
            FROM ScanHistory {where} ORDER BY ScannedAt DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        return rows

    HEADERS = ["ID","Patient","Age","Gender",
               "Pred1","Conf1","Pred2","Conf2","Pred3","Conf3",
               "Diagnosis","Severity","Recommendation","Time"]

    def exp_all(self, fmt):
        path = self._save(fmt, "AllScans")
        self._write(path, self.HEADERS, self._fetch(), fmt)

    def exp_today(self, fmt):
        today = datetime.now().strftime("%Y-%m-%d")
        path  = self._save(fmt, f"TodayScans_{today}")
        self._write(path, self.HEADERS,
                    self._fetch(f"WHERE CAST(ScannedAt AS DATE)='{today}'"), fmt)

    def exp_critical(self, fmt):
        path = self._save(fmt, "CriticalCases")
        self._write(path, self.HEADERS,
                    self._fetch("WHERE SeverityLevel IN ('Critical','High')"), fmt)

    def exp_summary(self, fmt):
        path = self._save(fmt, "DiagnosisSummary")
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT FinalDiagnosis,
                   COUNT(*) AS TotalCases,
                   AVG(Confidence1) AS AvgConfidence,
                   SeverityLevel
            FROM ScanHistory
            GROUP BY FinalDiagnosis, SeverityLevel
            ORDER BY TotalCases DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        self._write(path,
                    ["Diagnosis","Total Cases","Avg Confidence","Severity"],
                    rows, fmt)

    def open_detector(self):
        self.root.destroy()
        import detector
        detector.launch()

    def logout(self):
        if messagebox.askyesno("Logout", "Return to login?"):
            self.root.destroy()
            import main_app
            main_app.launch()


def launch():
    root = tk.Tk()
    Dashboard(root)
    root.mainloop()

if __name__ == "__main__":
    launch()