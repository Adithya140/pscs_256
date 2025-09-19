from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
import base64
import uuid
import logging
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, Optional
from dotenv import load_dotenv
import json
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configure Gemini API
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY environment variable not set")
    raise ValueError("GOOGLE_API_KEY environment variable is required")

genai.configure(api_key=api_key)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Enhanced session storage with automatic cleanup
class SessionManager:
    def __init__(self, cleanup_interval: int = 300):  # 5 minutes
        self.sessions: Dict[str, dict] = {}
        self.cleanup_interval = cleanup_interval
        self.start_cleanup_thread()
    
    def start_cleanup_thread(self):
        """Start background thread for session cleanup"""
        def cleanup_expired_sessions():
            while True:
                try:
                    current_time = datetime.now()
                    expired_sessions = []
                    
                    for session_id, session_data in self.sessions.items():
                        if current_time - session_data['last_activity'] > timedelta(hours=1):
                            expired_sessions.append(session_id)
                    
                    for session_id in expired_sessions:
                        del self.sessions[session_id]
                        logger.info(f"Cleaned up expired session: {session_id}")
                    
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    logger.error(f"Error in session cleanup: {e}")
                    time.sleep(self.cleanup_interval)
        
        cleanup_thread = threading.Thread(target=cleanup_expired_sessions, daemon=True)
        cleanup_thread.start()
    
    def create_session(self, image_data: dict) -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'image': image_data,
            'history': [],
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'message_count': 0
        }
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session data"""
        session = self.sessions.get(session_id)
        if session:
            session['last_activity'] = datetime.now()
        return session
    
    def update_session_history(self, session_id: str, user_message: dict, model_response: dict):
        """Update session history"""
        if session_id in self.sessions:
            self.sessions[session_id]['history'].extend([user_message, model_response])
            self.sessions[session_id]['last_activity'] = datetime.now()
            self.sessions[session_id]['message_count'] += 1
    
    def get_session_stats(self) -> dict:
        """Get session statistics"""
        return {
            'active_sessions': len(self.sessions),
            'total_messages': sum(s['message_count'] for s in self.sessions.values())
        }

# Initialize session manager
session_manager = SessionManager()

# Initialize Gemini models with error handling
def get_gemini_model():
    """Get Gemini model with fallback options"""
    models_to_try = [
        'gemini-1.5-flash-latest',
        'gemini-1.5-flash',
        'gemini-1.5-pro-latest',
        'gemini-pro-vision'
    ]
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            logger.info(f"Using Gemini model: {model_name}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load model {model_name}: {e}")
            continue
    
    raise Exception("No Gemini models available")

# Initialize model
try:
    model = get_gemini_model()
except Exception as e:
    logger.error(f"Failed to initialize Gemini model: {e}")
    model = None

# Enhanced safety settings
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

# Generation configuration for better responses
generation_config = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 2048,
}

def validate_image_file(file) -> tuple[bool, str]:
    """Validate uploaded image file"""
    if not file or file.filename == '':
        return False, "No file selected"
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff'}
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if file_ext not in allowed_extensions:
        return False, f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
    
    # Check MIME type
    if not file.content_type or not file.content_type.startswith('image/'):
        return False, "Invalid image file"
    
    return True, "Valid image file"

def process_image_data(image_file) -> dict:
    """Process and encode image data"""
    try:
        image_bytes = image_file.read()
        
        # Reset file pointer for potential re-reading
        image_file.seek(0)
        
        # Encode to base64
        image_data = base64.b64encode(image_bytes).decode('utf-8')
        
        return {
            'mime_type': image_file.content_type,
            'data': image_bytes,
            'data_b64': image_data,
            'filename': secure_filename(image_file.filename),
            'size': len(image_bytes)
        }
    except Exception as e:
        logger.error(f"Error processing image data: {e}")
        raise ValueError("Failed to process image data")

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_available': model is not None,
        'session_stats': session_manager.get_session_stats(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/start_chat', methods=['POST'])
def start_chat():
    """Handle initial image upload and first prompt"""
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        if 'prompt' not in request.form:
            return jsonify({'error': 'No prompt provided'}), 400
        
        image_file = request.files['image']
        prompt_text = request.form['prompt'].strip()
        
        if not prompt_text:
            return jsonify({'error': 'Prompt cannot be empty'}), 400
        
        if len(prompt_text) > 2000:
            return jsonify({'error': 'Prompt too long (max 2000 characters)'}), 400
        
        # Validate image
        is_valid, error_message = validate_image_file(image_file)
        if not is_valid:
            return jsonify({'error': error_message}), 400
        
        # Check model availability
        if not model:
            return jsonify({'error': 'AI model not available. Please try again later.'}), 503
        
        # Process image
        try:
            image_data = process_image_data(image_file)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        # Create session
        session_id = session_manager.create_session(image_data)
        
        # Prepare content for Gemini
        user_message = {
            'role': 'user', 
            'parts': [
                {
                    'inline_data': {
                        'mime_type': image_data['mime_type'],
                        'data': image_data['data_b64']
                    }
                },
                {'text': prompt_text}
            ]
        }
        
        # Generate response
        try:
            response = model.generate_content(
                [user_message],
                safety_settings=safety_settings,
                generation_config=generation_config
            )
            
            if not response.text:
                if response.candidates and response.candidates[0].finish_reason:
                    reason = response.candidates[0].finish_reason
                    if reason == "SAFETY":
                        return jsonify({'error': 'Content blocked due to safety concerns. Please try a different prompt.'}), 400
                    else:
                        return jsonify({'error': f'Response generation failed: {reason}'}), 500
                else:
                    return jsonify({'error': 'No response generated. Please try again.'}), 500
            
            model_message = {'role': 'model', 'parts': [{'text': response.text}]}
            
            # Update session
            session_manager.update_session_history(session_id, user_message, model_message)
            
            logger.info(f"Successfully started chat for session {session_id}")
            
            return jsonify({
                'session_id': session_id,
                'response': response.text
            })
            
        except Exception as e:
            # Clean up session on failure
            if session_id in session_manager.sessions:
                del session_manager.sessions[session_id]
            
            logger.error(f"Error generating content: {e}")
            return jsonify({'error': 'Failed to generate response. Please try again.'}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error in start_chat: {e}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle subsequent text chat messages"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        session_id = data.get('session_id')
        prompt_text = data.get('prompt', '').strip()
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        if not prompt_text:
            return jsonify({'error': 'Prompt cannot be empty'}), 400
        
        if len(prompt_text) > 2000:
            return jsonify({'error': 'Message too long (max 2000 characters)'}), 400
        
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found or expired. Please start a new conversation.'}), 404
        
        # Check model availability
        if not model:
            return jsonify({'error': 'AI model not available. Please try again later.'}), 503
        
        # Rate limiting: max 50 messages per session
        if session.get('message_count', 0) >= 50:
            return jsonify({'error': 'Message limit reached for this session. Please start a new conversation.'}), 429
        
        # Prepare message
        user_message = {'role': 'user', 'parts': [{'text': prompt_text}]}
        
        # Generate response with full context
        try:
            # Build conversation history for context
            conversation_history = []
            
            # Add initial image context if it's the first few messages
            if session.get('message_count', 0) < 5:
                conversation_history.append({
                    'role': 'user',
                    'parts': [
                        {
                            'inline_data': {
                                'mime_type': session['image']['mime_type'],
                                'data': session['image']['data_b64']
                            }
                        },
                        {'text': '[Image context for ongoing conversation]'}
                    ]
                })
            
            # Add recent history (last 10 messages)
            recent_history = session['history'][-10:] if len(session['history']) > 10 else session['history']
            conversation_history.extend(recent_history)
            
            # Add current message
            conversation_history.append(user_message)
            
            response = model.generate_content(
                conversation_history,
                safety_settings=safety_settings,
                generation_config=generation_config
            )
            
            if not response.text:
                if response.candidates and response.candidates[0].finish_reason:
                    reason = response.candidates[0].finish_reason
                    if reason == "SAFETY":
                        return jsonify({'error': 'Message blocked due to safety concerns. Please try a different message.'}), 400
                    else:
                        return jsonify({'error': f'Response generation failed: {reason}'}), 500
                else:
                    return jsonify({'error': 'No response generated. Please try again.'}), 500
            
            model_message = {'role': 'model', 'parts': [{'text': response.text}]}
            
            # Update session
            session_manager.update_session_history(session_id, user_message, model_message)
            
            return jsonify({'response': response.text})
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return jsonify({'error': 'Failed to generate response. Please try again.'}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error in chat: {e}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Handle voice input transcription for initial prompt"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if not audio_file or audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Check model availability
        if not model:
            return jsonify({'error': 'AI model not available. Please try again later.'}), 503
        
        try:
            # Prepare transcription request
            transcription_prompt = "Please transcribe the following audio accurately. Only return the transcribed text, no additional commentary:"
            
            response = model.generate_content([
                transcription_prompt,
                {
                    "mime_type": audio_file.content_type,
                    "data": base64.b64encode(audio_file.read()).decode()
                }
            ])
            
            if not response.text:
                return jsonify({'error': 'Failed to transcribe audio. Please try again.'}), 500
            
            # Clean up the transcription (remove extra whitespace, etc.)
            transcript = response.text.strip()
            
            if len(transcript) < 2:
                return jsonify({'error': 'Audio too short or unclear. Please try again.'}), 400
            
            logger.info(f"Successfully transcribed audio: {len(transcript)} characters")
            
            return jsonify({'transcript': transcript})
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return jsonify({'error': 'Failed to process audio. Please try again.'}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error in transcribe: {e}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

@app.route('/chat_voice', methods=['POST'])
def chat_voice():
    """Handle voice input for ongoing chat"""
    try:
        session_id = request.form.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if not audio_file or audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found or expired. Please start a new conversation.'}), 404
        
        # Check model availability
        if not model:
            return jsonify({'error': 'AI model not available. Please try again later.'}), 503
        
        # Rate limiting
        if session.get('message_count', 0) >= 50:
            return jsonify({'error': 'Message limit reached for this session. Please start a new conversation.'}), 429
        
        try:
            # Process audio file
            audio_bytes = audio_file.read()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            if len(audio_bytes) == 0:
                return jsonify({'error': 'Audio file is empty'}), 400
            
            # First, transcribe the audio
            transcription_response = model.generate_content([
                "Please transcribe the following audio accurately. Only return the transcribed text:",
                {
                    "mime_type": audio_file.content_type,
                    "data": audio_b64
                }
            ])
            
            if not transcription_response.text:
                return jsonify({'error': 'Failed to transcribe audio. Please try again.'}), 500
            
            transcript = transcription_response.text.strip()
            
            if len(transcript) < 2:
                return jsonify({'error': 'Audio too short or unclear. Please try again.'}), 400
            
            # Now process the transcribed text as a regular chat message
            user_message = {'role': 'user', 'parts': [{'text': transcript}]}
            
            # Build conversation history
            conversation_history = []
            
            # Add image context for recent messages
            if session.get('message_count', 0) < 5:
                conversation_history.append({
                    'role': 'user',
                    'parts': [
                        {
                            'inline_data': {
                                'mime_type': session['image']['mime_type'],
                                'data': session['image']['data_b64']
                            }
                        },
                        {'text': '[Image context for ongoing conversation]'}
                    ]
                })
            
            # Add recent history
            recent_history = session['history'][-10:] if len(session['history']) > 10 else session['history']
            conversation_history.extend(recent_history)
            
            # Add current transcribed message
            conversation_history.append(user_message)
            
            # Generate AI response
            response = model.generate_content(
                conversation_history,
                safety_settings=safety_settings,
                generation_config=generation_config
            )
            
            if not response.text:
                if response.candidates and response.candidates[0].finish_reason:
                    reason = response.candidates[0].finish_reason
                    if reason == "SAFETY":
                        return jsonify({'error': 'Voice message blocked due to safety concerns. Please try a different message.'}), 400
                    else:
                        return jsonify({'error': f'Response generation failed: {reason}'}), 500
                else:
                    return jsonify({'error': 'No response generated. Please try again.'}), 500
            
            model_message = {'role': 'model', 'parts': [{'text': response.text}]}
            
            # Update session
            session_manager.update_session_history(session_id, user_message, model_message)
            
            logger.info(f"Successfully processed voice message for session {session_id}")
            
            return jsonify({
                'transcript': transcript,
                'response': response.text
            })
            
        except Exception as e:
            logger.error(f"Error processing voice chat: {e}")
            return jsonify({'error': 'Failed to process voice message. Please try again.'}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error in chat_voice: {e}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

@app.route('/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a specific session"""
    try:
        if session_id in session_manager.sessions:
            del session_manager.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return jsonify({'message': 'Session deleted successfully'})
        else:
            return jsonify({'error': 'Session not found'}), 404
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        return jsonify({'error': 'Failed to delete session'}), 500

@app.route('/sessions', methods=['GET'])
def get_sessions():
    """Get session statistics (for admin/debugging)"""
    try:
        stats = session_manager.get_session_stats()
        return jsonify({
            'active_sessions': stats['active_sessions'],
            'total_messages': stats['total_messages'],
            'sessions': [
                {
                    'id': session_id,
                    'created_at': session_data['created_at'].isoformat(),
                    'last_activity': session_data['last_activity'].isoformat(),
                    'message_count': session_data['message_count'],
                    'image_filename': session_data['image'].get('filename', 'unknown')
                }
                for session_id, session_data in session_manager.sessions.items()
            ]
        })
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        return jsonify({'error': 'Failed to get session information'}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed_error(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(413)
def payload_too_large_error(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(429)
def ratelimit_handler(error):
    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error. Please try again later.'}), 500

@app.errorhandler(503)
def service_unavailable_error(error):
    return jsonify({'error': 'Service temporarily unavailable. Please try again later.'}), 503

# Request logging middleware
@app.before_request
def log_request_info():
    """Log incoming requests"""
    if request.endpoint != 'health':  # Don't log health checks
        logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def log_response_info(response):
    """Log outgoing responses"""
    if request.endpoint != 'health':  # Don't log health check responses
        logger.info(f"Response: {response.status_code} for {request.method} {request.path}")
    return response

# Context processors for templates
@app.context_processor
def inject_app_info():
    """Inject app information into templates"""
    return {
        'app_name': 'Enhanced Image Chat',
        'version': '2.0.0',
        'model_available': model is not None
    }

if __name__ == '__main__':
    # Production-ready configuration
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Enhanced Image Chat server on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Model available: {model is not None}")
    
    # In production, use a proper WSGI server like gunicorn
    if debug:
        app.run(
            host='0.0.0.0',
            port=port,
            debug=True,
            threaded=True
        )
    else:
        # For production deployment
        from waitress import serve
        logger.info("Running with Waitress WSGI server")
        serve(
            app,
            host='0.0.0.0',
            port=port,
            threads=4,
            connection_limit=100,
            cleanup_interval=30
        )
