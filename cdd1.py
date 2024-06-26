import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
data = pd.read_csv('cdd.csv')
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your model
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# Create the GUI
class FraudDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fraud Detection App")
        
        # Main Frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas
        self.canvas = tk.Canvas(self.main_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Scrollable Frame
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Input Frame
        self.input_frame = ttk.LabelFrame(self.scrollable_frame, text="Enter Feature Values")
        self.input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")    

        # Feature Labels and Entry Boxes
        self.feature_names = X.columns.tolist()
        self.entry_vars = []
        for i, feature in enumerate(self.feature_names):
            ttk.Label(self.input_frame, text=feature).grid(row=i, column=0, padx=5, pady=5, sticky="e")
            entry_var = tk.StringVar()
            ttk.Entry(self.input_frame, textvariable=entry_var).grid(row=i, column=1, padx=5, pady=5)
            self.entry_vars.append(entry_var)

        # Prediction Button
        ttk.Button(self.scrollable_frame, text="Predict", command=self.predict).grid(row=1, column=0, padx=10, pady=10)

        # Output Frame
        self.output_frame = ttk.LabelFrame(self.scrollable_frame, text="Prediction Result")
        self.output_frame.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.result_text = ScrolledText(self.output_frame, wrap=tk.WORD, width=40, height=10)
        self.result_text.pack(padx=5, pady=5)
        self.result_text.config(state=tk.DISABLED)

    def predict(self):
        # Extract user input
        user_input = [float(entry_var.get()) for entry_var in self.entry_vars]

        # Predict using the trained model
        predicted_class = random_forest_model.predict([user_input])
        result = "Fraud" if predicted_class[0] == 1 else "Not Fraud"
        
        # Display result
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"The predicted class is: {result}")
        self.result_text.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = FraudDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
