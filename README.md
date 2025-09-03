
# CONVERSATIONAL IMAGE RECOGNITION CHATBOT
An AI-powered web app that lets you upload an image, ask questions about it, and receive intelligent, contextual responses — powered by Google's Gemini Vision model.

---

## 📸 Overview
This is an interactive image analysis platform where you can:
- Upload any image (JPG, PNG, etc.)
- Ask natural language questions about the image
- Get AI-generated answers in real-time
- Continue chatting with context preserved

It combines **computer vision** with **conversational AI** to make image understanding seamless and interactive.

---

## 🚀 Features
- **Drag & Drop Upload**: Quickly load your image into the app
- **Prompt-Based Interaction**: Ask the AI anything about the uploaded image
- **Chat History Context**: Keeps your conversation relevant to the current image
- **Responsive UI**: Works on desktop and mobile devices
- **Powered by Gemini Vision**: State-of-the-art multimodal AI

---

## 🛠️ Project Structure
```

pscs_256/
├── app.py          # Flask backend serving the app and handling API calls
├── static/         # Static assets (CSS, JS, images, etc.)
└── templates/
└── index.html  # Frontend UI template

````

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Adithya140/pscs_256.git
cd pscs_256
````

### 2️⃣ Set up a Python environment

```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If `requirements.txt` doesn't exist yet, install manually:

```bash
pip install flask google-generativeai
```

---

## 🔑 API Key Setup

This app uses **Google Generative AI** (Gemini Vision).
You’ll need to:

1. Get an API key from [API KEYS](https://aistudio.google.com/)
2. Store it as an environment variable:

   ```bash
   export GOOGLE_API_KEY="your_api_key_here"    # Mac/Linux
   set GOOGLE_API_KEY="your_api_key_here"       # Windows (cmd)
   $env:GOOGLE_API_KEY="your_api_key_here"      # Windows (PowerShell)
   ```

---

## ▶️ Running the App

```bash
python app.py
```

By default, the app will be available at:
**[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 💡 Usage

1. Open the app in your browser.
2. Drag and drop or select an image to upload.
3. Enter your prompt/question in the text box.
4. Click **🚀 Start Conversation**.
5. Continue chatting — the AI remembers the image and your previous questions.

---

## 📷 Example Use Cases

* Identify objects, text, or scenes in an image
* Ask for descriptions or summaries
* Extract details or compare elements
* Generate captions or creative interpretations


---

## 🙌 Credits

* **Backend**: [Flask](https://flask.palletsprojects.com/)
* **Frontend**: HTML, CSS, JavaScript

