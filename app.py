from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, session, abort, send_from_directory
from groq import Groq
import os
from PIL import Image, ImageFilter
import io
import json
import base64
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from dotenv import load_dotenv
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import secrets
import re
import uuid
import logging
from functools import wraps
import jwt
import hashlib
import sqlite3

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask uygulamasını ilk satırda oluştur
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(24))

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///atlasai.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# File upload configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'txt', 'py', 'js', 'java', 'cpp', 'c', 'html', 'css'}

# Email configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True') == 'True'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', 'noreply@atlasai.com')

# Initialize extensions
db = SQLAlchemy(app)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
mail = Mail(app)

# Rate Limiter
limiter = Limiter(
    app=app,
    key_func=lambda: current_user.id if current_user.is_authenticated else get_remote_address(),
    default_limits=["200 per day", "50 per hour"]
)

# Groq API Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# System prompt
SYSTEM_PROMPT = """Sen Atlas Design tarafından geliştirilen Türkçe destekli bir yapay zeka asistansın. 
Tüm cevaplarını Türkçe ver. Kısa, net ve anlaşılır ol. 
Kod yazman istenirse, ilgili programlama dilinde açıklamalı kod örnekleri ver.
Matematiksel işlemleri çöz.
Eğer soru anlaşılmazsa, açıklama iste."""

# Database Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(80), nullable=False)
    password_hash = db.Column(db.String(256))
    api_key = db.Column(db.String(64), unique=True)
    subscription_tier = db.Column(db.String(20), default='free')  # free, pro, premium
    subscription_expiry = db.Column(db.DateTime)
    requests_today = db.Column(db.Integer, default=0)
    requests_total = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_api_key(self):
        self.api_key = secrets.token_hex(32)
        return self.api_key

    def can_make_request(self):
        if self.subscription_tier == 'premium':
            return True
        elif self.subscription_tier == 'pro':
            return self.requests_today < 1000
        else:  # free
            return self.requests_today < 100

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    chat_id = db.Column(db.String(64), nullable=False)
    title = db.Column(db.String(200))
    content = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class FileUpload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255))
    file_type = db.Column(db.String(50))
    file_size = db.Column(db.Integer)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

class APIRequestLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    endpoint = db.Column(db.String(100))
    method = db.Column(db.String(10))
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    response_time = db.Column(db.Float)
    status_code = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class DocumentAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255))
    file_type = db.Column(db.String(50))
    file_size = db.Column(db.Integer)
    question = db.Column(db.Text)
    analysis = db.Column(db.Text)
    analyzed_at = db.Column(db.DateTime, default=datetime.utcnow)

class ImageAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_hash = db.Column(db.String(64))
    analysis = db.Column(db.Text)
    analyzed_at = db.Column(db.DateTime, default=datetime.utcnow)

class VoiceTranscription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    audio_filename = db.Column(db.String(255))
    transcript = db.Column(db.Text)
    language = db.Column(db.String(10))
    transcribed_at = db.Column(db.DateTime, default=datetime.utcnow)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Lütfen giriş yapın'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Middleware for request logging
@app.before_request
def before_request():
    if request.endpoint and request.endpoint != 'static':
        session['last_activity'] = datetime.utcnow().isoformat()

@app.after_request
def after_request(response):
    if request.endpoint and request.endpoint != 'static':
        try:
            log = APIRequestLog(
                user_id=current_user.id if current_user.is_authenticated else None,
                endpoint=request.endpoint,
                method=request.method,
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string,
                status_code=response.status_code,
                response_time=0.0
            )
            db.session.add(log)
            db.session.commit()
        except:
            pass
    return response

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_chat_id():
    return str(uuid.uuid4())

def check_api_key(api_key):
    if not api_key:
        return None
    user = User.query.filter_by(api_key=api_key).first()
    return user if user and user.is_active else None

# API Key decorator
def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if not api_key:
            return jsonify({'error': 'API anahtarı gerekli'}), 401
        
        user = check_api_key(api_key)
        if not user:
            return jsonify({'error': 'Geçersiz API anahtarı'}), 401
        
        if not user.can_make_request():
            return jsonify({'error': 'Günlük limit doldu'}), 429
        
        # Update request count
        user.requests_today += 1
        user.requests_total += 1
        db.session.commit()
        
        return f(user, *args, **kwargs)
    return decorated_function

# Admin required decorator
def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

# ==================== AUTHENTICATION APIs ====================

@app.route('/api/v1/register', methods=['POST'])
def api_register():
    """Kullanıcı kayıt API"""
    data = request.json
    email = data.get('email')
    name = data.get('name')
    password = data.get('password')
    
    if not all([email, name, password]):
        return jsonify({'error': 'Tüm alanlar zorunludur'}), 400
    
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Bu email zaten kayıtlı'}), 400
    
    user = User(email=email, name=name)
    user.set_password(password)
    user.generate_api_key()
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Kayıt başarılı',
        'api_key': user.api_key,
        'user': {
            'id': user.id,
            'email': user.email,
            'name': user.name,
            'subscription_tier': user.subscription_tier
        }
    }), 201

@app.route('/api/v1/login', methods=['POST'])
def api_login():
    """API login endpoint"""
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({'error': 'Geçersiz email veya şifre'}), 401
    
    if not user.is_active:
        return jsonify({'error': 'Hesap devre dışı'}), 403
    
    user.last_login = datetime.utcnow()
    db.session.commit()
    
    return jsonify({
        'success': True,
        'api_key': user.api_key,
        'user': {
            'id': user.id,
            'email': user.email,
            'name': user.name,
            'subscription_tier': user.subscription_tier
        }
    })

@app.route('/api/v1/profile', methods=['GET'])
@api_key_required
def api_profile(user):
    """Kullanıcı profili API"""
    return jsonify({
        'user': {
            'id': user.id,
            'email': user.email,
            'name': user.name,
            'subscription_tier': user.subscription_tier,
            'requests_today': user.requests_today,
            'requests_total': user.requests_total,
            'subscription_expiry': user.subscription_expiry.isoformat() if user.subscription_expiry else None
        }
    })

@app.route('/api/v1/refresh-api-key', methods=['POST'])
@api_key_required
def api_refresh_key(user):
    """Yeni API anahtarı oluştur"""
    new_key = user.generate_api_key()
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'API anahtarı yenilendi',
        'api_key': new_key
    })

# ==================== CHAT APIs ====================

@app.route('/api/v1/chat', methods=['POST'])
@api_key_required
def api_chat(user):
    """Ana chat API"""
    data = request.json
    message = data.get('message')
    chat_id = data.get('chat_id', generate_chat_id())
    model = data.get('model', 'qwen/qwen3-32b')
    
    if not message:
        return jsonify({'error': 'Mesaj boş olamaz'}), 400
    
    if not client:
        return jsonify({'error': 'AI servisi kullanılamıyor'}), 503
    
    try:
        # Get chat history if exists
        history = ChatHistory.query.filter_by(user_id=user.id, chat_id=chat_id).first()
        conversation_history = []
        
        if history:
            try:
                conversation_history = json.loads(history.content) if history.content else []
            except:
                conversation_history = []
        
        # Add user message to history
        conversation_history.append({"role": "user", "content": message})
        
        # Call Groq API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *conversation_history[-10:]  # Son 10 mesaj
            ],
            max_tokens=4096,
            temperature=0.6,
            top_p=0.95
        )
        
        ai_response = response.choices[0].message.content.strip()
        # Remove <think> blocks
        ai_response = re.sub(r'<think>.*?</think>', '', ai_response, flags=re.DOTALL).strip()
        
        # Add AI response to history
        conversation_history.append({"role": "assistant", "content": ai_response})
        
        # Save or update chat history
        if history:
            history.content = json.dumps(conversation_history)
            history.updated_at = datetime.utcnow()
        else:
            title = message[:50] + "..." if len(message) > 50 else message
            history = ChatHistory(
                user_id=user.id,
                chat_id=chat_id,
                title=title,
                content=json.dumps(conversation_history)
            )
            db.session.add(history)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'response': ai_response,
            'chat_id': chat_id,
            'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else None
        })
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': f'AI servis hatası: {str(e)}'}), 500

@app.route('/api/v1/chats', methods=['GET'])
@api_key_required
def api_get_chats(user):
    """Kullanıcının tüm chat'lerini getir"""
    chats = ChatHistory.query.filter_by(user_id=user.id)\
        .order_by(ChatHistory.updated_at.desc())\
        .limit(50)\
        .all()
    
    return jsonify({
        'chats': [{
            'id': chat.id,
            'chat_id': chat.chat_id,
            'title': chat.title,
            'created_at': chat.created_at.isoformat(),
            'updated_at': chat.updated_at.isoformat()
        } for chat in chats]
    })

@app.route('/api/v1/chats/<chat_id>', methods=['GET'])
@api_key_required
def api_get_chat(user, chat_id):
    """Belirli bir chat'i getir"""
    chat = ChatHistory.query.filter_by(user_id=user.id, chat_id=chat_id).first()
    if not chat:
        return jsonify({'error': 'Chat bulunamadı'}), 404
    
    try:
        content = json.loads(chat.content) if chat.content else []
    except:
        content = []
    
    return jsonify({
        'chat': {
            'id': chat.id,
            'chat_id': chat.chat_id,
            'title': chat.title,
            'content': content,
            'created_at': chat.created_at.isoformat(),
            'updated_at': chat.updated_at.isoformat()
        }
    })

@app.route('/api/v1/chats/<chat_id>', methods=['DELETE'])
@api_key_required
def api_delete_chat(user, chat_id):
    """Chat sil"""
    chat = ChatHistory.query.filter_by(user_id=user.id, chat_id=chat_id).first()
    if not chat:
        return jsonify({'error': 'Chat bulunamadı'}), 404
    
    db.session.delete(chat)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Chat silindi'})

# ==================== CODE EXECUTION APIs ====================

@app.route('/api/v1/code/execute', methods=['POST'])
@api_key_required
def api_execute_code(user):
    """Kod çalıştırma API (Güvenli sandbox)"""
    data = request.json
    code = data.get('code')
    language = data.get('language', 'python')
    
    if not code:
        return jsonify({'error': 'Kod gerekli'}), 400
    
    # Safety checks
    blacklist = ['import os', 'import sys', '__import__', 'eval', 'exec', 'open(', 'subprocess']
    for item in blacklist:
        if item in code.lower():
            return jsonify({'error': 'Güvenlik nedeniyle bu kod çalıştırılamaz'}), 403
    
    try:
        if language == 'python':
            # Safe execution with limited globals/locals
            allowed_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'bool': bool,
                    'sum': sum,
                    'min': min,
                    'max': max,
                    'abs': abs,
                    'round': round,
                }
            }
            
            # Capture output
            import io
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                exec(code, allowed_globals, {})
                output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout
            
            return jsonify({
                'success': True,
                'output': output.strip() or 'Kod çalıştırıldı (çıktı yok)',
                'language': language
            })
            
        elif language == 'javascript':
            # For JavaScript, we would need a Node.js backend
            return jsonify({
                'error': 'JavaScript desteği yakında eklenecek',
                'code': code
            }), 501
            
        else:
            return jsonify({'error': f'Desteklenmeyen dil: {language}'}), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'language': language
        }), 500

@app.route('/api/v1/code/explain', methods=['POST'])
@api_key_required
def api_explain_code(user):
    """Kod açıklama API"""
    data = request.json
    code = data.get('code')
    language = data.get('language')
    
    if not code:
        return jsonify({'error': 'Kod gerekli'}), 400
    
    prompt = f"Aşağıdaki {language} kodunu açıkla:\n\n```{language}\n{code}\n```\n\nAçıklamayı Türkçe yap."
    
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {"role": "system", "content": "Sen bir kod açıklama uzmanısın. Kodları satır satır Türkçe açıklarsın."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2048,
            temperature=0.4
        )
        
        explanation = response.choices[0].message.content.strip()
        
        return jsonify({
            'success': True,
            'explanation': explanation,
            'language': language
        })
        
    except Exception as e:
        return jsonify({'error': f'Açıklama hatası: {str(e)}'}), 500

# ==================== FILE UPLOAD APIs ====================

@app.route('/api/v1/upload', methods=['POST'])
@api_key_required
def api_upload_file(user):
    """Dosya yükleme API"""
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'İzin verilmeyen dosya türü'}), 400
    
    # Create uploads directory if not exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Secure filename and save
    original_filename = file.filename
    filename = secure_filename(f"{user.id}_{datetime.utcnow().timestamp()}_{original_filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Save to database
    file_upload = FileUpload(
        user_id=user.id,
        filename=filename,
        original_filename=original_filename,
        file_type=file.content_type,
        file_size=os.path.getsize(filepath)
    )
    db.session.add(file_upload)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'file': {
            'id': file_upload.id,
            'filename': filename,
            'original_filename': original_filename,
            'file_type': file.content_type,
            'file_size': file_upload.file_size,
            'uploaded_at': file_upload.uploaded_at.isoformat(),
            'download_url': f"/api/v1/files/{filename}"
        }
    })

@app.route('/api/v1/files', methods=['GET'])
@api_key_required
def api_get_files(user):
    """Kullanıcının dosyalarını listele"""
    files = FileUpload.query.filter_by(user_id=user.id)\
        .order_by(FileUpload.uploaded_at.desc())\
        .all()
    
    return jsonify({
        'files': [{
            'id': file.id,
            'filename': file.filename,
            'original_filename': file.original_filename,
            'file_type': file.file_type,
            'file_size': file.file_size,
            'uploaded_at': file.uploaded_at.isoformat(),
            'download_url': f"/api/v1/files/{file.filename}"
        } for file in files]
    })

@app.route('/api/v1/files/<filename>', methods=['GET'])
@api_key_required
def api_download_file(user, filename):
    """Dosya indirme API"""
    file = FileUpload.query.filter_by(filename=filename, user_id=user.id).first()
    if not file:
        return jsonify({'error': 'Dosya bulunamadı'}), 404
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'Dosya bulunamadı'}), 404
    
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename,
        as_attachment=True,
        download_name=file.original_filename
    )

# ==================== IMAGE ANALYSIS APIs ====================

@app.route('/api/v1/image/analyze', methods=['POST'])
@api_key_required
def api_analyze_image(user):
    """Görsel analiz API"""
    try:
        data = request.json
        image_data = data.get('image')
        question = data.get('question', 'Bu görselde ne görüyorsun?')
        
        if not image_data:
            return jsonify({'error': 'Görsel verisi gerekli'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        
        # Save image temporarily
        image_path = f"uploads/temp_{user.id}_{datetime.utcnow().timestamp()}.jpg"
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        try:
            # Analyze image with AI
            prompt = f"""
            Kullanıcı bir görsel yükledi ve şu soruyu soruyor: "{question}"
            
            Görsel analizi yap ve aşağıdaki başlıklarda detaylı açıklama yap:
            1. Görselde ne olduğu
            2. Renkler, kompozisyon, ışık
            3. Varsa metin içeriği
            4. Tahmini konu veya bağlam
            5. Kullanıcının sorusuna özel cevap
            
            Cevaplarını Türkçe ve detaylı ver.
            """
            
            response = client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[
                    {"role": "system", "content": "Sen bir görsel analiz uzmanısın. Görselleri detaylı şekilde analiz edip Türkçe açıklarsın."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.5
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # Save to database
            image_hash = hashlib.md5(image_bytes).hexdigest()
            img_analysis = ImageAnalysis(
                user_id=user.id,
                image_hash=image_hash,
                analysis=analysis
            )
            db.session.add(img_analysis)
            db.session.commit()
            
            # Clean up temp file
            if os.path.exists(image_path):
                os.remove(image_path)
            
            return jsonify({
                'success': True,
                'analysis': analysis,
                'image_size': len(image_bytes)
            })
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(image_path):
                os.remove(image_path)
            raise e
            
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        return jsonify({'error': f'Görsel analiz hatası: {str(e)}'}), 500

@app.route('/api/v1/image/ocr', methods=['POST'])
@api_key_required
def api_ocr_image(user):
    """Görselden metin okuma (OCR) API"""
    try:
        # This would require Tesseract OCR
        # For now, return a placeholder
        return jsonify({
            'success': True,
            'text': 'OCR özelliği yakında eklenecek. Lütfen daha sonra tekrar deneyin.',
            'language': 'tr'
        })
    except Exception as e:
        return jsonify({'error': f'OCR hatası: {str(e)}'}), 500

# ==================== VOICE APIs ====================

@app.route('/api/v1/voice/synthesize', methods=['POST'])
@api_key_required
def api_synthesize_voice(user):
    """Metinden sese API"""
    try:
        data = request.json
        text = data.get('text')
        language = data.get('language', 'tr-TR')
        
        if not text:
            return jsonify({'error': 'Metin gerekli'}), 400
        
        # Note: Groq doesn't have TTS yet
        # This is a placeholder for future implementation
        return jsonify({
            'success': True,
            'message': 'Ses sentezi özelliği yakında eklenecek',
            'text': text,
            'language': language
        })
        
    except Exception as e:
        return jsonify({'error': f'Ses sentezi hatası: {str(e)}'}), 500

@app.route('/api/v1/voice/transcribe', methods=['POST'])
@api_key_required
def api_transcribe_voice(user):
    """Sesten metne API"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'Ses dosyası gerekli'}), 400
        
        audio_file = request.files['audio']
        language = request.form.get('language', 'tr-TR')
        
        # Save audio file
        audio_path = f"uploads/audio_{user.id}_{datetime.utcnow().timestamp()}.wav"
        audio_file.save(audio_path)
        
        try:
            # Note: This would require a speech recognition service
            # For now, return a placeholder
            transcript = "Ses tanıma özelliği yakında eklenecek. Lütfen daha sonra tekrar deneyin."
            
            # Save to database
            voice_transcript = VoiceTranscription(
                user_id=user.id,
                audio_filename=os.path.basename(audio_path),
                transcript=transcript,
                language=language
            )
            db.session.add(voice_transcript)
            db.session.commit()
            
            # Clean up
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return jsonify({
                'success': True,
                'transcript': transcript,
                'language': language
            })
            
        except Exception as e:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            raise e
            
    except Exception as e:
        return jsonify({'error': f'Ses tanıma hatası: {str(e)}'}), 500

# ==================== DOCUMENT ANALYSIS APIs ====================

@app.route('/api/v1/document/analyze', methods=['POST'])
@api_key_required
def api_analyze_document(user):
    """Doküman analiz API"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Dosya gerekli'}), 400
        
        file = request.files['file']
        question = request.form.get('question', 'Bu dosyada ne yazıyor?')
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Desteklenmeyen dosya formatı'}), 400
        
        # Save file
        filename = secure_filename(f"{user.id}_{datetime.utcnow().timestamp()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read file content based on type
            content = ""
            
            if filename.lower().endswith('.pdf'):
                # Extract text from PDF
                try:
                    import PyPDF2
                    with open(filepath, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        for page in pdf_reader.pages:
                            content += page.extract_text() + "\n"
                except:
                    content = "PDF içeriği okunamadı"
                    
            elif filename.lower().endswith(('.txt', '.md', '.json', '.xml', '.html')):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
            elif filename.lower().endswith(('.docx', '.doc')):
                # Would need python-docx library
                content = "Word dokümanı - içerik okuma özelliği yakında eklenecek"
                
            else:
                content = "Bu dosya formatı desteklenmiyor"
            
            # Analyze content with AI
            prompt = f"""
            Kullanıcı bir dosya yükledi ve şu soruyu soruyor: "{question}"
            
            Dosya içeriği:
            ```text
            {content[:4000]}  # Limit content size
            ```
            
            Dosyayı analiz et ve kullanıcının sorusunu cevapla.
            Cevaplarını Türkçe ve detaylı ver.
            """
            
            response = client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[
                    {"role": "system", "content": "Sen bir doküman analiz uzmanısın. Dosya içeriklerini analiz edip Türkçe özetlersin."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.5
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # Save to database
            doc_analysis = DocumentAnalysis(
                user_id=user.id,
                filename=filename,
                original_filename=file.filename,
                file_type=file.content_type,
                file_size=os.path.getsize(filepath),
                question=question,
                analysis=analysis
            )
            db.session.add(doc_analysis)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'analysis': analysis,
                'file_info': {
                    'filename': filename,
                    'original_name': file.filename,
                    'size': os.path.getsize(filepath),
                    'type': file.content_type
                }
            })
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e
            
    except Exception as e:
        logger.error(f"Document analysis error: {str(e)}")
        return jsonify({'error': f'Doküman analiz hatası: {str(e)}'}), 500

# ==================== ADMIN APIs ====================

@app.route('/api/v1/admin/users', methods=['GET'])
@admin_required
def api_admin_users():
    """Tüm kullanıcıları listele (Admin only)"""
    users = User.query.order_by(User.created_at.desc()).all()
    
    return jsonify({
        'users': [{
            'id': user.id,
            'email': user.email,
            'name': user.name,
            'subscription_tier': user.subscription_tier,
            'requests_today': user.requests_today,
            'requests_total': user.requests_total,
            'created_at': user.created_at.isoformat(),
            'last_login': user.last_login.isoformat() if user.last_login else None,
            'is_active': user.is_active,
            'is_admin': user.is_admin
        } for user in users]
    })

@app.route('/api/v1/admin/stats', methods=['GET'])
@admin_required
def api_admin_stats():
    """Sistem istatistikleri (Admin only)"""
    total_users = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    total_requests = db.session.query(db.func.sum(User.requests_total)).scalar() or 0
    today_requests = db.session.query(db.func.sum(User.requests_today)).scalar() or 0
    
    # Recent activity
    recent_logs = APIRequestLog.query\
        .order_by(APIRequestLog.created_at.desc())\
        .limit(100)\
        .all()
    
    return jsonify({
        'stats': {
            'total_users': total_users,
            'active_users': active_users,
            'total_requests': total_requests,
            'today_requests': today_requests,
            'subscription_distribution': {
                'free': User.query.filter_by(subscription_tier='free').count(),
                'pro': User.query.filter_by(subscription_tier='pro').count(),
                'premium': User.query.filter_by(subscription_tier='premium').count()
            }
        },
        'recent_activity': [{
            'id': log.id,
            'user_id': log.user_id,
            'endpoint': log.endpoint,
            'method': log.method,
            'status_code': log.status_code,
            'created_at': log.created_at.isoformat()
        } for log in recent_logs]
    })

@app.route('/api/v1/admin/user/<int:user_id>', methods=['PUT'])
@admin_required
def api_admin_update_user(user_id):
    """Kullanıcı güncelle (Admin only)"""
    user = User.query.get_or_404(user_id)
    data = request.json
    
    if 'subscription_tier' in data:
        user.subscription_tier = data['subscription_tier']
    
    if 'subscription_expiry' in data:
        user.subscription_expiry = datetime.fromisoformat(data['subscription_expiry'])
    
    if 'is_active' in data:
        user.is_active = bool(data['is_active'])
    
    if 'is_admin' in data:
        user.is_admin = bool(data['is_admin'])
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Kullanıcı güncellendi',
        'user': {
            'id': user.id,
            'email': user.email,
            'name': user.name,
            'subscription_tier': user.subscription_tier,
            'is_active': user.is_active,
            'is_admin': user.is_admin
        }
    })

# ==================== UTILITY APIs ====================

@app.route('/api/v1/models', methods=['GET'])
@api_key_required
def api_get_models(user):
    """Kullanılabilir AI modellerini listele"""
    try:
        # Note: Groq API doesn't have a models endpoint yet
        # This is a static list for now
        models = [
            'qwen/qwen3-32b',
            'llama3-70b-8192',
            'mixtral-8x7b-32768',
            'gemma-7b-it'
        ]
        
        return jsonify({
            'models': models,
            'default': 'qwen/qwen3-32b'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/health', methods=['GET'])
def api_health():
    """Sistem sağlık kontrolü"""
    try:
        # Check database
        db.session.execute('SELECT 1')
        
        # Check Groq API
        groq_status = False
        if client:
            try:
                response = client.chat.completions.create(
                    model="qwen/qwen3-32b",
                    messages=[{"role": "user", "content": "Merhaba"}],
                    max_tokens=1
                )
                groq_status = True
            except:
                groq_status = False
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'services': {
                'database': 'healthy',
                'groq_api': 'healthy' if groq_status else 'unavailable',
                'cache': 'healthy'
            },
            'uptime': 'N/A'  # Would need uptime tracking
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/v1/usage', methods=['GET'])
@api_key_required
def api_usage(user):
    """Kullanım istatistikleri"""
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    today_requests = APIRequestLog.query\
        .filter_by(user_id=user.id)\
        .filter(APIRequestLog.created_at >= today_start)\
        .count()
    
    return jsonify({
        'usage': {
            'subscription_tier': user.subscription_tier,
            'requests_today': user.requests_today,
            'requests_total': user.requests_total,
            'requests_today_db': today_requests,
            'subscription_expiry': user.subscription_expiry.isoformat() if user.subscription_expiry else None,
            'limits': {
                'free': 100,
                'pro': 1000,
                'premium': 'unlimited'
            }
        }
    })

# ==================== DASHBOARD APIs ====================

@app.route('/api/v1/dashboard/stats', methods=['GET'])
@login_required
def api_dashboard_stats():
    """Dashboard istatistikleri"""
    user = current_user
    
    # Daily request chart data (last 7 days)
    import datetime
    dates = []
    requests = []
    
    for i in range(6, -1, -1):
        date = datetime.datetime.utcnow() - datetime.timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        
        count = APIRequestLog.query.filter(
            APIRequestLog.user_id == user.id,
            db.func.date(APIRequestLog.created_at) == date.date()
        ).count()
        
        dates.append(date_str)
        requests.append(count)
    
    return jsonify({
        'stats': {
            'total_chats': ChatHistory.query.filter_by(user_id=user.id).count(),
            'total_files': FileUpload.query.filter_by(user_id=user.id).count(),
            'total_requests': user.requests_total,
            'today_requests': user.requests_today,
            'subscription_tier': user.subscription_tier
        },
        'chart_data': {
            'dates': dates,
            'requests': requests
        }
    })

# ==================== WEB ROUTES ====================

@app.route('/')
@login_required
def index():
    return render_template('index.html', user=current_user)

@app.route('/login')
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/login/custom', methods=['POST'])
def login_custom():
    data = request.json
    email = data.get('email')
    password = data.get('password', '')  # Password support

    if not email:
        return jsonify({'success': False, 'error': 'Email gerekli'}), 400
    
    # Check if it's a simple Gmail login (for backward compatibility)
    if not password and email.endswith('@gmail.com'):
        name = email.split('@')[0]
        user = User.query.filter_by(email=email).first()
        if not user:
            user = User(email=email, name=name)
            user.generate_api_key()
            db.session.add(user)
        
        user.last_login = datetime.utcnow()
        db.session.commit()
        login_user(user)
        
        return jsonify({'success': True})
    
    # Normal password login
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        user.last_login = datetime.utcnow()
        db.session.commit()
        login_user(user)
        return jsonify({'success': True})
    
    return jsonify({'success': False, 'error': 'Geçersiz email veya şifre'}), 401

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('login'))

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'Mesaj boş olamaz'}), 400
    
    if not current_user.can_make_request():
        return jsonify({'error': 'Günlük limit doldu'}), 429
    
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=4096,
            temperature=0.6,
            top_p=0.95
        )
        
        ai_response = response.choices[0].message.content.strip()
        ai_response = re.sub(r'<think>.*?</think>', '', ai_response, flags=re.DOTALL).strip()
        
        # Update request count
        current_user.requests_today += 1
        current_user.requests_total += 1
        db.session.commit()
        
        return jsonify({'response': ai_response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_chat', methods=['POST'])
def save_chat():
    data = request.json
    chat_id = data.get('chat_id')
    chat_content = data.get('chat_content')
    if not chat_id or not chat_content:
        return jsonify({'error': 'Chat ID ve içerik gerekli'}), 400

    try:
        with open(f'chats/{chat_id}.json', 'w', encoding='utf-8') as f:
            json.dump({'chat_id': chat_id, 'content': chat_content, 'timestamp': data.get('timestamp')}, f, ensure_ascii=False)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/load_chats', methods=['GET'])
def load_chats():
    try:
        chats = []
        if os.path.exists('chats'):
            for file in os.listdir('chats'):
                if file.endswith('.json'):
                    with open(f'chats/{file}', 'r', encoding='utf-8') as f:
                        chat_data = json.load(f)
                        chats.append(chat_data)
        return jsonify({'chats': chats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint bulunamadı'}), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Sunucu hatası'}), 500
    return render_template('500.html'), 500

@app.errorhandler(429)
def ratelimit_error(error):
    return jsonify({
        'error': 'Rate limit aşıldı',
        'message': 'Lütfen bir süre bekleyin veya premium plana yükselin'
    }), 429

# ==================== INITIALIZATION ====================

def init_db():
    """Veritabanını başlat"""
    with app.app_context():
        db.create_all()
        
        # Create admin user if not exists
        admin_email = os.getenv('ADMIN_EMAIL')
        admin_password = os.getenv('ADMIN_PASSWORD')
        
        if admin_email and admin_password:
            admin = User.query.filter_by(email=admin_email).first()
            if not admin:
                admin = User(
                    email=admin_email,
                    name='Admin',
                    is_admin=True
                )
                admin.set_password(admin_password)
                admin.generate_api_key()
                db.session.add(admin)
                db.session.commit()
                print(f"Admin user created: {admin_email}")

def cleanup_old_files():
    """Eski dosyaları temizle"""
    with app.app_context():
        try:
            # Delete files older than 7 days
            cutoff = datetime.utcnow() - timedelta(days=7)
            old_files = FileUpload.query.filter(FileUpload.uploaded_at < cutoff).all()
            
            for file in old_files:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
                db.session.delete(file)
            
            db.session.commit()
            logger.info(f"Cleaned up {len(old_files)} old files")
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")


# ==================== MAIN ====================

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('chats', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    # Initialize database
    init_db()
    
    # Run app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)