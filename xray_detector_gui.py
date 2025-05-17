import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
import os

# Load model
model = load_model('model/tb_model.h5')
classes = ['normal', 'tb_low', 'tb_medium', 'tb_severe']
symptoms_list = [
    'chills', 'vomiting', 'fatigue', 'weight_loss', 'cough', 'high_fever', 'breathlessness',
    'sweating', 'loss_of_appetite', 'mild_fever', 'yellowing_of_eyes', 'swelled_lymph_nodes',
    'malaise', 'phlegm', 'chest_pain', 'blood_in_sputum'
]

# GUI
app = tk.Tk()
app.title("🧬 Tuberculosis Detection System")
app.geometry("880x650")
app.configure(bg="#15151f")

img_label = tk.Label(app, bg="#15151f")
img_label.pack(pady=10)

selected_symptoms = {}
image_path = None

def upload_image():
    global image_path
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if path:
        image_path = path
        img = Image.open(path).resize((300, 300))
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

def predict():
    if not image_path:
        messagebox.showerror("⚠️ Error", "Please upload an X-ray image first.")
        return

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    index = np.argmax(pred)
    prediction = classes[index]
    confidence = round(pred[index] * 100, 2)

    selected = [symptom for symptom, var in selected_symptoms.items() if var.get()]
    symptom_percentage = round((len(selected) / len(symptoms_list)) * 100, 2)

    if prediction == 'normal':
        result = "No Tuberculosis Detected ✅"
        severity = "-"
    else:
        result = "Tuberculosis Detected ❗"
        severity = prediction.split('_')[1].capitalize()

    overall_risk = round((confidence + symptom_percentage) / 2, 2)

    precautions = ""
    if prediction != 'normal':
        precautions = (
            "\n- Wear a mask\n"
            "- Avoid close contact\n"
            "- Complete TB medication\n"
            "- Maintain hygiene"
        )

    consult = "YES 🩺" if overall_risk > 50 else "Recommended 📌"

    result_text = (
        "📋 DIAGNOSIS SUMMARY\n"
        "──────────────────────────────\n"
        f"🩺 Result: {result}\n"
        f"🫁 Image Prediction Confidence: {confidence}%\n"
        f"📊 Symptom Match: {symptom_percentage}%\n"
        f"📌 TB Severity: {severity}\n"
        f"⚠️ Overall Risk: {overall_risk}%\n"
        f"👨‍⚕️ Consult Doctor: {consult}\n"
        f"🧾 Precautions:{precautions}\n"
        "──────────────────────────────"
    )

    messagebox.showinfo("Diagnosis Result", result_text)

# Upload button
tk.Button(app, text="📤 Upload Chest X-ray", command=upload_image,
          bg="#4CAF50", fg="white", padx=12, pady=6,
          font=("Helvetica", 11, "bold")).pack(pady=10)

# Symptoms checkboxes
symptom_frame = tk.LabelFrame(app, text="Select Symptoms", bg="#1f1f2e", fg="white", font=("Helvetica", 14, "bold"))
symptom_frame.pack(pady=15)

for idx, symptom in enumerate(symptoms_list):
    var = tk.IntVar()
    cb = tk.Checkbutton(
        symptom_frame,
        text=symptom.replace('_', ' ').title(),
        variable=var,
        bg="#1f1f2e",
        fg="white",
        selectcolor="#2e2e3e",
        font=("Arial", 11),
        activebackground="#1f1f2e",
        activeforeground="white"
    )
    cb.grid(row=idx//2, column=idx%2, sticky='w', padx=15, pady=2)
    selected_symptoms[symptom] = var

# Predict button
tk.Button(app, text="🔍 Run Diagnosis", command=predict,
          bg="#2196F3", fg="white", padx=12, pady=6,
          font=("Helvetica", 11, "bold")).pack(pady=20)

app.mainloop()
