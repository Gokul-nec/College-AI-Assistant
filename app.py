import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from llm_provider import CollegeChatbot

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///college_assistant.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {'pdf', 'txt'}

db = SQLAlchemy(app)
CORS(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='student')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    sessions = db.relationship('ChatSession', backref='user', lazy=True, cascade='all, delete-orphan')
    documents = db.relationship('Document', backref='uploaded_by', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {'id': self.id, 'username': self.username, 'email': self.email, 'role': self.role, 'created_at': self.created_at.isoformat()}

class Document(db.Model):
    __tablename__ = 'documents'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)
    file_size = db.Column(db.Integer)
    uploaded_by_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    category = db.Column(db.String(100), default='General')
    description = db.Column(db.Text)

    def to_dict(self):
        return {'id': self.id, 'filename': self.filename, 'file_type': self.file_type, 'file_size': self.file_size, 'uploaded_at': self.uploaded_at.isoformat(), 'category': self.category, 'description': self.description}

class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(255), default='New Chat')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    messages = db.relationship('ChatMessage', backref='session', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        return {'id': self.id, 'session_id': self.session_id, 'title': self.title, 'created_at': self.created_at.isoformat(), 'updated_at': self.updated_at.isoformat(), 'message_count': len(self.messages)}

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    sources = db.Column(db.JSON)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {'id': self.id, 'role': self.role, 'content': self.content, 'sources': self.sources, 'timestamp': self.timestamp.isoformat()}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_user_from_session():
    user_id = session.get('user_id')
    if user_id:
        return User.query.get(user_id)
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    user = get_user_from_session()
    if not user:
        return redirect('/')
    return render_template('dashboard.html')

@app.route('/chat')
def chat_page():
    user = get_user_from_session()
    if not user:
        return redirect('/')
    return render_template('chat.html')

@app.route('/admin')
def admin_page():
    user = get_user_from_session()
    if not user or user.role != 'admin':
        return redirect('/')
    return render_template('admin.html')

@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json() or {}
        username = (data.get('username') or '').strip()
        email = (data.get('email') or '').strip()
        password = data.get('password')

        if not username or not email or not password:
            return jsonify({'error': 'Username, email and password are required'}), 400
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 409
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already exists'}), 409

        user = User(username=username, email=email, role=data.get('role', 'student'))
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        session['user_id'] = user.id
        session['username'] = user.username

        return jsonify({'message': 'User registered successfully', 'user': user.to_dict()}), 201
    except Exception as exc:
        db.session.rollback()
        return jsonify({'error': str(exc)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json() or {}
        username = (data.get('username') or '').strip()
        password = data.get('password')

        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400
        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            return jsonify({'error': 'Invalid username or password'}), 401

        session['user_id'] = user.id
        session['username'] = user.username
        return jsonify({'message': 'Login successful', 'user': user.to_dict()}), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/api/auth/profile', methods=['GET'])
def get_profile():
    user = get_user_from_session()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    return jsonify(user.to_dict()), 200

@app.route('/api/documents/upload', methods=['POST'])
def upload_document():
    user = get_user_from_session()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF and TXT files are supported'}), 400

        filename = secure_filename(file.filename)
        unique_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(save_path)

        document = Document(
            filename=file.filename,
            file_path=save_path,
            file_type=filename.rsplit('.', 1)[1].lower(),
            file_size=os.path.getsize(save_path),
            uploaded_by_id=user.id,
            category=request.form.get('category', 'General'),
            description=request.form.get('description', '')
        )
        db.session.add(document)
        db.session.commit()
        return jsonify({'message': 'Document uploaded successfully', 'document': document.to_dict()}), 201
    except Exception as exc:
        db.session.rollback()
        return jsonify({'error': str(exc)}), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    try:
        docs = Document.query.filter_by(is_active=True).all()
        return jsonify([doc.to_dict() for doc in docs]), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

@app.route('/api/documents/<int:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    user = get_user_from_session()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    doc = Document.query.get(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    if doc.uploaded_by_id != user.id and user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    doc.is_active = False
    db.session.commit()
    return jsonify({'message': 'Document deleted'}), 200

@app.route('/api/chat/sessions', methods=['GET'])
def get_sessions():
    user = get_user_from_session()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    sessions = ChatSession.query.filter_by(user_id=user.id, is_active=True).all()
    return jsonify([s.to_dict() for s in sessions]), 200

@app.route('/api/chat/sessions/new', methods=['POST'])
def create_session():
    user = get_user_from_session()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    data = request.get_json() or {}
    chat_session = ChatSession(
        user_id=user.id,
        session_id=os.urandom(16).hex(),
        title=data.get('title', 'New Chat')
    )
    db.session.add(chat_session)
    db.session.commit()
    return jsonify({'message': 'Session created', 'session': chat_session.to_dict()}), 201

@app.route('/api/chat/sessions/<int:session_id>', methods=['GET'])
def get_session_messages(session_id):
    user = get_user_from_session()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    chat_session = ChatSession.query.get(session_id)
    if not chat_session or chat_session.user_id != user.id:
        return jsonify({'error': 'Session not found'}), 404
    messages = ChatMessage.query.filter_by(session_id=chat_session.id).order_by(ChatMessage.timestamp.asc()).all()
    return jsonify({'session': chat_session.to_dict(), 'messages': [item.to_dict() for item in messages]}), 200

@app.route('/api/chat/message', methods=['POST'])
def chat_message():
    user = get_user_from_session()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        data = request.get_json() or {}
        message = (data.get('message') or '').strip()
        session_id = data.get('session_id')
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        chat_session = ChatSession.query.get(session_id)
        if not chat_session or chat_session.user_id != user.id:
            return jsonify({'error': 'Invalid session'}), 400
        user_msg = ChatMessage(session_id=chat_session.id, role='user', content=message)
        db.session.add(user_msg)
        db.session.flush()
        if not hasattr(app, 'chatbot'):
            app.chatbot = CollegeChatbot()
        response_text, sources = app.chatbot.get_response(message)
        assistant_msg = ChatMessage(
            session_id=chat_session.id,
            role='assistant',
            content=response_text,
            sources=sources
        )
        db.session.add(assistant_msg)
        chat_session.updated_at = datetime.utcnow()
        db.session.commit()
        return jsonify({
            'user_message': user_msg.to_dict(),
            'assistant_message': assistant_msg.to_dict()
        }), 200
    except Exception as exc:
        db.session.rollback()
        return jsonify({'error': str(exc)}), 500

@app.errorhandler(404)
def not_found(_):
    return jsonify({'error': 'Route not found'}), 404

@app.errorhandler(500)
def server_error(_):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', email='admin@college.edu', role='admin')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print('Default admin created: admin / admin123')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
