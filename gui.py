import os
import subprocess
import platform

# Check if DISPLAY is set, and if not, start Xvfb (only for non-Windows systems)
if platform.system() != "Windows" and not os.environ.get("DISPLAY"):
    try:
        print("$DISPLAY not set. Starting Xvfb...")
        subprocess.Popen(
            ["Xvfb", ":100", "-screen", "0", "1024x768x16"]
        )  # Use display :100 instead of :99
        os.environ["DISPLAY"] = ":100"  # Set DISPLAY to use Xvfb
    except FileNotFoundError:
        print(
            "Error: Xvfb is not installed. Please install it using 'sudo apt-get install xvfb'."
        )
        exit(1)

import tkinter as tk
from tkinter import ttk, messagebox
import queue
import logging
from trade_simulator import RealTradeSimulator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import time
import numpy as np


# Configure logging to GUI
class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, msg + "\n")
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")


class TradeSimulatorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Trade Simulator")
        self.root.geometry("1000x800")

        self.output_queue = queue.Queue(maxsize=1000)
        self.simulator = RealTradeSimulator(output_queue=self.output_queue)

        self.input_vars = {}
        self.output_labels_vars = {}
        self.mid_prices = deque(maxlen=100)
        self.volatilities = deque(maxlen=100)

        # Latency tracking
        self.ui_update_latencies = deque(maxlen=1000)

        self._create_widgets()
        self.update_output_display()
        self.check_queue_interval = 100
        self._check_queue()

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)

        # Input Panel
        input_frame = ttk.LabelFrame(
            paned_window, text="Input Parameters", padding="10"
        )
        paned_window.add(input_frame, weight=1)

        input_params = {
            "Exchange": ["OKX"],
            "Spot Asset": "BTC-USDT-SWAP",
            "Order Type": ["market"],
            "Quantity (USD)": "100.0",
            "Volatility (Market Param)": "0.5",
            "Fee Tier": "1",
            "Risk Aversion (λ)": "1e-6",
            "Total Execution Time (H)": "1.0",
            "Number of Trades (N)": "10",
        }

        row = 0
        for label_text, default_value in input_params.items():
            lbl = ttk.Label(input_frame, text=f"{label_text}:")
            lbl.grid(row=row, column=0, sticky=tk.W, pady=2)
            if isinstance(default_value, list):
                var = tk.StringVar(value=default_value[0])
                entry = ttk.Combobox(
                    input_frame,
                    textvariable=var,
                    values=default_value,
                    state="readonly",
                )
            else:
                var = tk.StringVar(value=default_value)
                entry = ttk.Entry(input_frame, textvariable=var, width=25)
            entry.grid(row=row, column=1, sticky=tk.EW, pady=2, padx=5)
            self.input_vars[label_text] = var
            row += 1

        ttk.Button(
            input_frame, text="Apply Inputs", command=self.apply_input_changes
        ).grid(row=row, column=0, columnspan=2, pady=10)
        row += 1

        self.start_button = ttk.Button(
            input_frame, text="Start Simulation", command=self.start_simulation
        )
        self.start_button.grid(row=row, column=0, pady=5, sticky=tk.EW)
        self.stop_button = ttk.Button(
            input_frame,
            text="Stop Simulation",
            command=self.stop_simulation,
            state=tk.DISABLED,
        )
        self.stop_button.grid(row=row, column=1, pady=5, sticky=tk.EW)
        row += 1

        self.simulator_status_var = tk.StringVar(value="Simulator Stopped")
        ttk.Label(input_frame, textvariable=self.simulator_status_var).grid(
            row=row, column=0, columnspan=2, pady=5
        )

        input_frame.columnconfigure(1, weight=1)

        # Output Panel
        output_frame = ttk.LabelFrame(
            paned_window, text="Processed Output Values", padding="10"
        )
        paned_window.add(output_frame, weight=2)

        output_params = [
            "Timestamp",
            "Expected Slippage (%)",
            "Expected Fees (USD)",
            "Expected Market Impact (USD)",
            "Net Cost (USD)",
            "Maker Proportion",
            "Taker Proportion",
            "Internal Latency (ms)",
            "Current Volatility (raw tick)",
            "Order Book Imbalance",
        ]

        row = 0
        for label_text in output_params:
            lbl_name = ttk.Label(output_frame, text=f"{label_text}:")
            lbl_name.grid(row=row, column=0, sticky=tk.W, pady=2)
            var_value = tk.StringVar(value="N/A")
            lbl_value = ttk.Label(output_frame, textvariable=var_value, anchor="e")
            lbl_value.grid(row=row, column=1, sticky=tk.EW, pady=2, padx=5)
            self.output_labels_vars[label_text] = var_value
            row += 1

        output_frame.columnconfigure(1, weight=1)

        # Plot Panel
        plot_frame = ttk.LabelFrame(main_frame, text="Real-Time Metrics", padding="10")
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 4))
        (self.line1,) = self.ax1.plot([], [], label="Mid Price")
        self.ax1.set_xlabel("Time")
        self.ax1.set_ylabel("Mid Price")
        self.ax1.legend()
        (self.line2,) = self.ax2.plot([], [], label="Volatility")
        self.ax2.set_xlabel("Time")
        self.ax2.set_ylabel("Volatility")
        self.ax2.legend()
        canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Log Panel
        log_frame = ttk.LabelFrame(main_frame, text="Simulator Logs", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, height=5, state="disabled")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        logging.getLogger().addHandler(TextHandler(self.log_text))

        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Add latency stats to the status bar
        ttk.Button(
            main_frame, text="Latency Stats", command=self.show_latency_stats
        ).pack(side=tk.BOTTOM, fill=tk.X)

    def apply_input_changes(self):
        """Apply input changes with validation."""
        start_time = time.perf_counter()
        try:
            new_config = {
                "exchange": self.input_vars["Exchange"].get(),
                "spot_asset": self.input_vars["Spot Asset"].get(),
                "order_type": self.input_vars["Order Type"].get(),
                "quantity_usd": float(self.input_vars["Quantity (USD)"].get()),
                "volatility_annualized_config": float(
                    self.input_vars["Volatility (Market Param)"].get()
                ),
                "fee_tier": int(self.input_vars["Fee Tier"].get()),
                "ac_risk_aversion_lambda": float(
                    self.input_vars["Risk Aversion (λ)"].get()
                ),
                "ac_total_time_to_execute_H_hours": float(
                    self.input_vars["Total Execution Time (H)"].get()
                ),
                "ac_num_trades_N": int(self.input_vars["Number of Trades (N)"].get()),
            }
            for key, value in new_config.items():
                if (
                    key
                    in [
                        "quantity_usd",
                        "volatility_annualized_config",
                        "ac_total_time_to_execute_H_hours",
                        "ac_num_trades_N",
                    ]
                    and value <= 0
                ):
                    raise ValueError(f"{key} must be positive")
                if key == "fee_tier" and value not in [1, 2, 3, 4]:
                    raise ValueError("Fee Tier must be 1, 2, 3, or 4")
            self.simulator.update_config(new_config)
            self.status_var.set("Input parameters applied")
        except ValueError as e:
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Input Error", f"Invalid input: {e}")
        finally:
            self.ui_update_latencies.append((time.perf_counter() - start_time) * 1000)

    def start_simulation(self):
        """Start the simulation with VPN reminder."""
        if messagebox.askokcancel(
            "VPN Check", "Ensure VPN is active for OKX WebSocket connection. Proceed?"
        ):
            self.apply_input_changes()
            self.simulator.start()
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.simulator_status_var.set("Simulator Running")
            self.status_var.set("Simulation started")

    def stop_simulation(self):
        """Stop the simulation."""
        self.simulator.stop()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.simulator_status_var.set("Simulator Stopped")
        self.status_var.set("Simulation stopped")

    def _check_queue(self):
        """Check queue for new data."""
        start_time = time.perf_counter()
        try:
            for _ in range(10):  # Process up to 10 messages
                data = self.output_queue.get_nowait()
                self.update_output_display(data)
        except queue.Empty:
            pass
        finally:
            self.ui_update_latencies.append((time.perf_counter() - start_time) * 1000)
            self.root.after(self.check_queue_interval, self._check_queue)

    def update_output_display(self, latest_data=None):
        """Update output labels and plots."""
        if latest_data:
            for key, var in self.output_labels_vars.items():
                var.set(latest_data.get(key, "N/A"))
            try:
                mid_price = float(latest_data["Expected Slippage (%)"])
                volatility = float(latest_data["Current Volatility (raw tick)"])
                self.mid_prices.append(mid_price)
                self.volatilities.append(volatility)
                self.line1.set_data(range(len(self.mid_prices)), list(self.mid_prices))
                self.line2.set_data(
                    range(len(self.volatilities)), list(self.volatilities)
                )
                for ax in [self.ax1, self.ax2]:
                    ax.relim()
                    ax.autoscale_view()
                self.fig.canvas.draw()
            except (ValueError, KeyError):
                pass
        else:
            for key, var in self.output_labels_vars.items():
                if not var.get():
                    var.set("N/A")

    def get_latency_stats(self):
        """Return UI latency statistics."""
        if not self.ui_update_latencies:
            return {"avg": 0.0, "max": 0.0, "min": 0.0, "count": 0}
        latencies = list(self.ui_update_latencies)
        return {
            "avg": np.mean(latencies),
            "max": np.max(latencies),
            "min": np.min(latencies),
            "count": len(latencies),
        }

    def show_latency_stats(self):
        """Display latency statistics."""
        stats = self.simulator.get_latency_stats()
        messagebox.showinfo(
            "Latency Statistics",
            f"Average: {stats['avg']:.2f} ms\nMax: {stats['max']:.2f} ms\nMin: {stats['min']:.2f} ms\nCount: {stats['count']}",
        )

    def on_closing(self):
        """Handle window close."""
        self.status_var.set("Closing application")
        self.stop_simulation()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    ui = TradeSimulatorUI(root)
    root.protocol("WM_DELETE_WINDOW", ui.on_closing)
    root.mainloop()
