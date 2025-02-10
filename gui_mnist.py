import tkinter as tk
from tkinter import ttk
import os

# Define the root directory containing your files
BASE_DIR = 'out'

class MNISTMetricsGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MNIST Metrics Viewer")
        self.configure(padx=10, pady=10)

        # Style for better visuals
        style = ttk.Style()
        style.configure("TLabel", font=("Arial", 12))
        style.configure("TCombobox", font=("Arial", 12))
        style.configure("TButton", font=("Arial", 12))

        # Framework selection (including 'tensorflow', 'pytorch', 'jax')
        self.framework_var = tk.StringVar()
        ttk.Label(self, text="Select Framework:").grid(row=0, column=0, sticky="w")
        self.framework_dropdown = ttk.Combobox(self, textvariable=self.framework_var, width=20)
        self.framework_dropdown['values'] = ['pytorch', 'tensorflow', 'jax']
        self.framework_dropdown.grid(row=0, column=1, pady=5, padx=5)

        # Show Metrics button
        self.show_button = ttk.Button(self, text="Show Metrics", command=self.show_metrics)
        self.show_button.grid(row=1, columnspan=2, pady=10)

        # Frame for displaying metrics in a structured format
        self.metrics_frame = ttk.Frame(self)
        self.metrics_frame.grid(row=2, columnspan=2, pady=5, padx=5, sticky="nsew")

    def show_metrics(self):
        # Clear previous metrics
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()

        # Get the selected framework
        framework = self.framework_var.get()

        # Construct the path to `output_0.txt` for the MNIST dataset
        file_path = os.path.join(BASE_DIR, framework, 'mnist', 'output_0.txt')

        try:
            with open(file_path, 'r') as file:
                data = file.read()
                # Parse metrics from `output_0.txt`
                metrics = self.parse_mnist_metrics(data)

                # Header row
                ttk.Label(self.metrics_frame, text="Metric", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", padx=5)
                ttk.Label(self.metrics_frame, text="Value", font=("Arial", 12, "bold")).grid(row=0, column=1, sticky="w", padx=5)

                # Display each metric
                row = 1
                for metric, value in metrics.items():
                    ttk.Label(self.metrics_frame, text=metric).grid(row=row, column=0, sticky="w", padx=5)
                    ttk.Label(self.metrics_frame, text=value).grid(row=row, column=1, sticky="w", padx=5)
                    row += 1
        except FileNotFoundError:
            ttk.Label(self.metrics_frame, text="Metrics file not available for MNIST.", font=("Arial", 12, "italic")).grid(row=0, columnspan=2, pady=5, padx=5)

    def parse_mnist_metrics(self, data):
        # Extract and rename metrics from `output_0.txt`
        metrics = {}
        for line in data.splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                # Map raw keys to more descriptive labels
                metric_name = {
                    "Accuracy": "Accuracy",
                    "F1 Score": "F1 Score",
                    "Training Time": "Training Time (s)",
                    "Inference Time": "Inference Time (s)",
                    "Memory Usage Increase": "Memory Usage (MB)"
                }.get(key.strip(), key.strip())  # Default to the key if not in the map
                metrics[metric_name] = value.strip()
        return metrics

# Run the application
if __name__ == "__main__":
    app = MNISTMetricsGUI()
    app.mainloop()
