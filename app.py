# app.py
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
import base64
import uuid

# Set up your Gemini API key from environment variables or paste it here directly
# It is highly recommended to use environment variables for security
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
genai.configure(api_key="AIzaSyACOj1SZ7F0TF4CvVhtyRwqXH1LC_fxRz4")
app = Flask(__name__)

# Dictionary to store chat history per session. This is a simple
# way to maintain state for a demonstration. For a production app,
# you would use a database or a more robust session management system.
sessions = {}

# Use the latest vision-capable model
model = genai.GenerativeModel('gemini-1.5-flash-latest')

@app.route('/')
def index():
    """Renders the main HTML page for the chatbot."""
    return render_template('index.html')

@app.route('/start_chat', methods=['POST'])
def start_chat():
    """
    Handles the initial image upload and first prompt.
    It creates a new session and returns the initial response.
    """
    if 'image' not in request.files or 'prompt' not in request.form:
        return jsonify({'error': 'Missing image or prompt'}), 400

    image_file = request.files['image']
    prompt_text = request.form['prompt']

    # Read the image file and encode it to base64
    image_bytes = image_file.read()
    image_data = base64.b64encode(image_bytes).decode('utf-8')
    mime_type = image_file.mimetype

    # Create a unique session ID for this conversation
    session_id = str(uuid.uuid4())
    
    # Store the image and initial chat history for this session
    sessions[session_id] = {
        'image': {
            'mime_type': mime_type,
            'data': image_bytes
        },
        'history': [
            {'role': 'user', 'parts': [
                {'inline_data': {'mime_type': mime_type, 'data': image_data}},
                {'text': prompt_text}
            ]}
        ]
    }
    
    try:
        # Generate the initial content
        response = model.generate_content(sessions[session_id]['history'])
        
        # Add the model's response to the history
        sessions[session_id]['history'].append({'role': 'model', 'parts': [{'text': response.text}]})
        
        return jsonify({
            'session_id': session_id,
            'response': response.text
        })
    except Exception as e:
        print(f"Error starting chat: {e}")
        del sessions[session_id]  # Clean up the session on failure
        return jsonify({'error': 'Failed to generate content.'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles subsequent chat messages within an existing session.
    It uses the stored image and chat history for context.
    """
    data = request.get_json()
    session_id = data.get('session_id')
    prompt_text = data.get('prompt')

    if not session_id or not prompt_text:
        return jsonify({'error': 'Missing session ID or prompt'}), 400
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 404

    try:
        # Append the new user message to the history
        sessions[session_id]['history'].append({'role': 'user', 'parts': [{'text': prompt_text}]})
        
        # Note: The image data from the initial request is already in the history,
        # so we don't need to pass it again. The model will use the full conversation
        # history for context.
        
        response = model.generate_content(sessions[session_id]['history'])
        
        # Append the model's new response to the history
        sessions[session_id]['history'].append({'role': 'model', 'parts': [{'text': response.text}]})

        return jsonify({'response': response.text})
    except Exception as e:
        print(f"Error in chat: {e}")
        return jsonify({'error': 'Failed to generate content.'}), 500

if __name__ == '__main__':
    app.run(debug=True)

