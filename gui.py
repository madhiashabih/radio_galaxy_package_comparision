import tkinter as tk
from tkinter import ttk
import os

# Define the root directory containing your files
BASE_DIR = 'out'

class MetricsGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Model Metrics Viewer")
        self.configure(padx=10, pady=10)
        
        # Set up style for better visuals
        style = ttk.Style()
        style.configure("TLabel", font=("Arial", 12))
        style.configure("TCombobox", font=("Arial", 12))
        style.configure("TButton", font=("Arial", 12))
        
        # Framework selection (excluding 'jax')
        self.framework_var = tk.StringVar()
        ttk.Label(self, text="Select Framework:").grid(row=0, column=0, sticky="w")
        self.framework_dropdown = ttk.Combobox(self, textvariable=self.framework_var, width=20)
        self.framework_dropdown['values'] = ['pytorch', 'tensorflow']  # Excluding 'jax'
        self.framework_dropdown.grid(row=0, column=1, pady=5, padx=5)
        
        # Dataset selection (excluding 'mnist')
        self.dataset_var = tk.StringVar()
        ttk.Label(self, text="Select Dataset:").grid(row=1, column=0, sticky="w")
        self.dataset_dropdown = ttk.Combobox(self, textvariable=self.dataset_var, width=20)
        self.dataset_dropdown['values'] = ['frdeep-f', 'mirabest']  # Excluding 'mnist'
        self.dataset_dropdown.grid(row=1, column=1, pady=5, padx=5)
        
        # Model selection
        self.model_var = tk.StringVar()
        ttk.Label(self, text="Select Model:").grid(row=2, column=0, sticky="w")
        self.model_dropdown = ttk.Combobox(self, textvariable=self.model_var, width=20)
        self.model_dropdown['values'] = ['convxpress', 'first_class', 'mcrgnet', 'toothless']
        self.model_dropdown.grid(row=2, column=1, pady=5, padx=5)
        
        # Show Metrics button
        self.show_button = ttk.Button(self, text="Show Metrics", command=self.show_metrics)
        self.show_button.grid(row=3, columnspan=2, pady=10)
        
        # Create a frame to display metrics in a structured format
        self.metrics_frame = ttk.Frame(self)
        self.metrics_frame.grid(row=4, columnspan=2, pady=5, padx=5, sticky="nsew")

    def show_metrics(self):
        # Clear previous metrics
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()
        
        # Define metric file names and display names
        metrics_files = {
            'accuracy.txt': 'Accuracy',
            'class_time.txt': 'Inference Time (s)',
            'f1.txt': 'F1 Score',
            'mem_use.txt': 'Memory Usage (MB)',
            'train_time.txt': 'Training Time (s)'
        }

        # Construct the path based on framework and dataset selection
        framework = self.framework_var.get()
        dataset = self.dataset_var.get()
        path = os.path.join(BASE_DIR, framework, dataset, self.model_var.get(), 'run_1')

        # Header row
        ttk.Label(self.metrics_frame, text="Metric", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(self.metrics_frame, text="Value", font=("Arial", 12, "bold")).grid(row=0, column=1, sticky="w", padx=5)

        # Populate metrics in a structured format
        row = 1
        for file_name, display_name in metrics_files.items():
            file_path = os.path.join(path, file_name)
            try:
                with open(file_path, 'r') as file:
                    value = file.read().strip()
            except FileNotFoundError:
                value = "Not available"
            
            # Display metric name and value in two columns
            ttk.Label(self.metrics_frame, text=display_name).grid(row=row, column=0, sticky="w", padx=5)
            ttk.Label(self.metrics_frame, text=value).grid(row=row, column=1, sticky="w", padx=5)
            row += 1

# Run the application
if __name__ == "__main__":
    app = MetricsGUI()
    app.mainloop()
