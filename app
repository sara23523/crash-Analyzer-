# app.py - Main Flask Backend for Crash Analyzer
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
import os
import cv2
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import json
import uuid
import threading
from werkzeug.utils import secure_filename
import hashlib
from ultralytics import YOLO
import torch
from pathlib import Path
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'crash-analyzer-secret-key-2024'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Initialize extensions
CORS(app, origins=['http://localhost:3000'])
jwt = JWTManager(app)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global variables for models
yolo_model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DatabaseManager:
    def __init__(self, db_path='crash_analyzer.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'investigator',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id TEXT PRIMARY KEY,
                user_id INTEGER,
                video_filename TEXT NOT NULL,
                video_path TEXT NOT NULL,
                status TEXT DEFAULT 'processing',
                results JSON,
                accuracy_score REAL,
                processing_time REAL,
                accident_type TEXT,
                responsible_vehicle TEXT,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create default admin user
        admin_password = hashlib.sha256('admin123'.encode()).hexdigest()
        cursor.execute('''
            INSERT OR IGNORE INTO users (username, password_hash, role) 
            VALUES (?, ?, ?)
        ''', ('admin', admin_password, 'admin'))
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)

# Initialize database
db_manager = DatabaseManager()

class CrashAnalyzer:
    def __init__(self):
        self.load_models()
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck in COCO
        
    def load_models(self):
        """Load YOLO model for vehicle detection"""
        global yolo_model
        try:
            # Try to load YOLOv8 model
            model_path = 'models/yolov8n.pt'
            if not os.path.exists(model_path):
                logger.info("Downloading YOLOv8 model...")
                yolo_model = YOLO('yolov8n.pt')
                yolo_model.save(model_path)
            else:
                yolo_model = YOLO(model_path)
            logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            yolo_model = None
    
    def preprocess_video(self, video_path, target_resolution=(1080, 720)):
        """Extract and preprocess frames from video"""
        frames = []
        timestamps = []
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame if needed
            height, width = frame.shape[:2]
            if width > target_resolution[0] or height > target_resolution[1]:
                scale = min(target_resolution[0]/width, target_resolution[1]/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            frames.append(frame)
            timestamps.append(frame_count / fps)
            frame_count += 1
        
        cap.release()
        return frames, timestamps, fps
    
    def detect_vehicles(self, frames):
        """Detect vehicles in frames using YOLO"""
        if yolo_model is None:
            return []
        
        detections = []
        for i, frame in enumerate(frames):
            try:
                results = yolo_model(frame, classes=self.vehicle_classes, verbose=False)
                frame_detections = []
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            
                            if conf > 0.5:  # Confidence threshold
                                frame_detections.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': float(conf),
                                    'class': cls,
                                    'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                                })
                
                detections.append({
                    'frame_id': i,
                    'vehicles': frame_detections
                })
            except Exception as e:
                logger.error(f"Error detecting vehicles in frame {i}: {e}")
                detections.append({'frame_id': i, 'vehicles': []})
        
        return detections
    
    def track_vehicles(self, detections):
        """Simple centroid-based vehicle tracking"""
        tracks = []
        active_tracks = {}
        track_id = 0
        max_distance = 100  # Maximum distance for track association
        
        for frame_detection in detections:
            frame_id = frame_detection['frame_id']
            vehicles = frame_detection['vehicles']
            
            # Update existing tracks
            updated_tracks = set()
            
            for vehicle in vehicles:
                center = vehicle['center']
                best_track = None
                min_distance = float('inf')
                
                # Find closest existing track
                for tid, track in active_tracks.items():
                    if len(track['positions']) > 0:
                        last_pos = track['positions'][-1]
                        distance = np.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
                        
                        if distance < min_distance and distance < max_distance:
                            min_distance = distance
                            best_track = tid
                
                if best_track is not None:
                    # Update existing track
                    active_tracks[best_track]['positions'].append(center)
                    active_tracks[best_track]['frames'].append(frame_id)
                    active_tracks[best_track]['bboxes'].append(vehicle['bbox'])
                    updated_tracks.add(best_track)
                else:
                    # Create new track
                    active_tracks[track_id] = {
                        'id': track_id,
                        'positions': [center],
                        'frames': [frame_id],
                        'bboxes': [vehicle['bbox']],
                        'class': vehicle['class']
                    }
                    updated_tracks.add(track_id)
                    track_id += 1
            
            # Remove tracks that weren't updated
            tracks_to_remove = []
            for tid in active_tracks:
                if tid not in updated_tracks:
                    # If track wasn't updated for 10 frames, consider it finished
                    if frame_id - active_tracks[tid]['frames'][-1] > 10:
                        tracks.append(active_tracks[tid].copy())
                        tracks_to_remove.append(tid)
            
            for tid in tracks_to_remove:
                del active_tracks[tid]
        
        # Add remaining active tracks
        for track in active_tracks.values():
            tracks.append(track)
        
        return tracks
    
    def analyze_collision(self, tracks, frames, timestamps):
        """Analyze potential collision between vehicles"""
        collision_events = []
        
        # Look for intersection of vehicle paths
        for i, track1 in enumerate(tracks):
            for j, track2 in enumerate(tracks[i+1:], i+1):
                # Check for collision between track1 and track2
                collision = self.detect_collision_between_tracks(track1, track2, timestamps)
                if collision:
                    collision_events.append(collision)
        
        return collision_events
    
    def detect_collision_between_tracks(self, track1, track2, timestamps):
        """Detect collision between two vehicle tracks"""
        # Find overlapping time frames
        common_frames = set(track1['frames']) & set(track2['frames'])
        
        if len(common_frames) < 5:  # Need at least 5 frames of overlap
            return None
        
        common_frames = sorted(list(common_frames))
        
        # Calculate distances between vehicles over time
        distances = []
        for frame in common_frames:
            idx1 = track1['frames'].index(frame)
            idx2 = track2['frames'].index(frame)
            
            pos1 = track1['positions'][idx1]
            pos2 = track2['positions'][idx2]
            
            distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            distances.append(distance)
        
        # Look for sudden decrease in distance (potential collision)
        min_distance = min(distances)
        if min_distance < 50:  # Very close proximity
            # Calculate velocities before collision
            collision_frame = common_frames[distances.index(min_distance)]
            
            vel1 = self.calculate_velocity(track1, collision_frame, timestamps)
            vel2 = self.calculate_velocity(track2, collision_frame, timestamps)
            
            return {
                'frame': collision_frame,
                'time': timestamps[collision_frame] if collision_frame < len(timestamps) else 0,
                'vehicle1': track1['id'],
                'vehicle2': track2['id'],
                'distance': min_distance,
                'velocity1': vel1,
                'velocity2': vel2,
                'severity': self.calculate_severity(vel1, vel2, min_distance)
            }
        
        return None
    
    def calculate_velocity(self, track, frame, timestamps):
        """Calculate velocity of a vehicle at a given frame"""
        try:
            frame_idx = track['frames'].index(frame)
            if frame_idx < 2:
                return 0
            
            # Use last 3 positions to calculate velocity
            pos_current = track['positions'][frame_idx]
            pos_prev = track['positions'][frame_idx - 1]
            
            time_current = timestamps[track['frames'][frame_idx]]
            time_prev = timestamps[track['frames'][frame_idx - 1]]
            
            dt = time_current - time_prev
            if dt == 0:
                return 0
            
            dx = pos_current[0] - pos_prev[0]
            dy = pos_current[1] - pos_prev[1]
            
            velocity = np.sqrt(dx**2 + dy**2) / dt
            return velocity
        except:
            return 0
    
    def calculate_severity(self, vel1, vel2, distance):
        """Calculate collision severity based on velocities and distance"""
        # Simple severity calculation
        combined_velocity = abs(vel1) + abs(vel2)
        
        if combined_velocity > 100 and distance < 30:
            return 'high'
        elif combined_velocity > 50 and distance < 40:
            return 'medium'
        else:
            return 'low'
    
    def determine_responsibility(self, collision_events, tracks):
        """Determine which vehicle is responsible for the accident"""
        if not collision_events:
            return None
        
        # Take the most severe collision
        main_collision = max(collision_events, key=lambda x: x['velocity1'] + x['velocity2'])
        
        # Simple rule-based responsibility determination
        vehicle1_id = main_collision['vehicle1']
        vehicle2_id = main_collision['vehicle2']
        vel1 = main_collision['velocity1']
        vel2 = main_collision['velocity2']
        
        # Basic rules:
        # 1. Higher velocity vehicle is more likely responsible
        # 2. Rear-end collisions: following vehicle is responsible
        # 3. Side collisions: analyze movement patterns
        
        if vel1 > vel2 * 1.5:
            responsible = vehicle1_id
            confidence = 0.8
        elif vel2 > vel1 * 1.5:
            responsible = vehicle2_id
            confidence = 0.8
        else:
            # More complex analysis needed - for now, assign based on position
            responsible = vehicle1_id if vehicle1_id < vehicle2_id else vehicle2_id
            confidence = 0.6
        
        return {
            'responsible_vehicle': responsible,
            'confidence': confidence,
            'reasoning': f"Based on velocity analysis and movement patterns",
            'collision_details': main_collision
        }

# Initialize analyzer
analyzer = CrashAnalyzer()

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User authentication endpoint"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, role FROM users WHERE username = ? AND password_hash = ?', 
                   (username, password_hash))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        access_token = create_access_token(identity={'user_id': user[0], 'role': user[1]})
        return jsonify({
            'access_token': access_token,
            'user': {'id': user[0], 'username': username, 'role': user[1]}
        })
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/upload', methods=['POST'])
@jwt_required()
def upload_video():
    """Upload video for analysis"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.webm'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({'error': 'Unsupported file format'}), 400
    
    # Generate unique filename
    analysis_id = str(uuid.uuid4())
    filename = secure_filename(f"{analysis_id}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(filepath)
        
        # Save to database
        user_identity = get_jwt_identity()
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_results (id, user_id, video_filename, video_path, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (analysis_id, user_identity['user_id'], file.filename, filepath, 'uploaded'))
        
        conn.commit()
        conn.close()
        
        # Start analysis in background
        threading.Thread(target=analyze_video_background, args=(analysis_id, filepath)).start()
        
        return jsonify({
            'analysis_id': analysis_id,
            'filename': file.filename,
            'status': 'uploaded'
        })
    
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({'error': 'Upload failed'}), 500

def analyze_video_background(analysis_id, video_path):
    """Background task to analyze uploaded video"""
    try:
        start_time = time.time()
        
        # Update status to processing
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE analysis_results SET status = ? WHERE id = ?', 
                      ('processing', analysis_id))
        conn.commit()
        conn.close()
        
        # Perform analysis
        logger.info(f"Starting analysis for {analysis_id}")
        
        # Step 1: Preprocess video
        frames, timestamps, fps = analyzer.preprocess_video(video_path)
        logger.info(f"Extracted {len(frames)} frames")
        
        # Step 2: Detect vehicles
        detections = analyzer.detect_vehicles(frames)
        logger.info(f"Detected vehicles in {len(detections)} frames")
        
        # Step 3: Track vehicles
        tracks = analyzer.track_vehicles(detections)
        logger.info(f"Found {len(tracks)} vehicle tracks")
        
        # Step 4: Analyze collisions
        collision_events = analyzer.analyze_collision(tracks, frames, timestamps)
        logger.info(f"Found {len(collision_events)} collision events")
        
        # Step 5: Determine responsibility
        responsibility = analyzer.determine_responsibility(collision_events, tracks)
        
        processing_time = time.time() - start_time
        
        # Prepare results
        results = {
            'video_info': {
                'total_frames': len(frames),
                'fps': fps,
                'duration': len(frames) / fps if fps > 0 else 0
            },
            'vehicle_tracks': len(tracks),
            'collision_events': collision_events,
            'responsibility_analysis': responsibility,
            'processing_time': processing_time
        }
        
        # Determine accident type and confidence
        accident_type = 'collision' if collision_events else 'no_accident'
        confidence_score = responsibility['confidence'] if responsibility else 0.5
        responsible_vehicle = responsibility['responsible_vehicle'] if responsibility else None
        
        # Update database with results
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE analysis_results 
            SET status = ?, results = ?, processing_time = ?, accident_type = ?, 
                responsible_vehicle = ?, confidence_score = ?
            WHERE id = ?
        ''', ('completed', json.dumps(results), processing_time, accident_type, 
              str(responsible_vehicle), confidence_score, analysis_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Analysis completed for {analysis_id}")
        
    except Exception as e:
        logger.error(f"Error analyzing video {analysis_id}: {e}")
        
        # Update status to error
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE analysis_results SET status = ? WHERE id = ?', 
                      ('error', analysis_id))
        conn.commit()
        conn.close()

@app.route('/api/analysis/<analysis_id>', methods=['GET'])
@jwt_required()
def get_analysis_result(analysis_id):
    """Get analysis results"""
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT status, results, accident_type, responsible_vehicle, 
               confidence_score, processing_time, created_at, video_filename
        FROM analysis_results WHERE id = ?
    ''', (analysis_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return jsonify({'error': 'Analysis not found'}), 404
    
    status, results_json, accident_type, responsible_vehicle, confidence_score, processing_time, created_at, video_filename = result
    
    response = {
        'analysis_id': analysis_id,
        'status': status,
        'video_filename': video_filename,
        'accident_type': accident_type,
        'responsible_vehicle': responsible_vehicle,
        'confidence_score': confidence_score,
        'processing_time': processing_time,
        'created_at': created_at
    }
    
    if results_json:
        response['results'] = json.loads(results_json)
    
    return jsonify(response)

@app.route('/api/analyses', methods=['GET'])
@jwt_required()
def get_user_analyses():
    """Get all analyses for current user"""
    user_identity = get_jwt_identity()
    
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, video_filename, status, accident_type, confidence_score, 
               processing_time, created_at
        FROM analysis_results 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', (user_identity['user_id'],))
    
    results = cursor.fetchall()
    conn.close()
    
    analyses = []
    for result in results:
        analyses.append({
            'analysis_id': result[0],
            'video_filename': result[1],
            'status': result[2],
            'accident_type': result[3],
            'confidence_score': result[4],
            'processing_time': result[5],
            'created_at': result[6]
        })
    
    return jsonify({'analyses': analyses})

@app.route('/api/report/<analysis_id>/pdf', methods=['GET'])
@jwt_required()
def generate_pdf_report(analysis_id):
    """Generate PDF report for analysis"""
    try:
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT video_filename, results, accident_type, responsible_vehicle, 
                   confidence_score, processing_time, created_at
            FROM analysis_results WHERE id = ?
        ''', (analysis_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return jsonify({'error': 'Analysis not found'}), 404
        
        video_filename, results_json, accident_type, responsible_vehicle, confidence_score, processing_time, created_at = result
        results = json.loads(results_json) if results_json else {}
        
        # Generate PDF
        pdf_filename = f"crash_report_{analysis_id}.pdf"
        pdf_path = os.path.join(app.config['RESULTS_FOLDER'], pdf_filename)
        
        c = canvas.Canvas(pdf_path, pagesize=letter)
        
        # Title
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, 750, "Crash Analysis Report")
        
        # Analysis details
        c.setFont("Helvetica", 12)
        y_pos = 700
        
        c.drawString(50, y_pos, f"Analysis ID: {analysis_id}")
        y_pos -= 20
        c.drawString(50, y_pos, f"Video File: {video_filename}")
        y_pos -= 20
        c.drawString(50, y_pos, f"Analysis Date: {created_at}")
        y_pos -= 20
        c.drawString(50, y_pos, f"Processing Time: {processing_time:.2f} seconds")
        y_pos -= 40
        
        # Results
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos, "Analysis Results:")
        y_pos -= 20
        
        c.setFont("Helvetica", 12)
        c.drawString(50, y_pos, f"Accident Type: {accident_type}")
        y_pos -= 20
        c.drawString(50, y_pos, f"Responsible Vehicle: {responsible_vehicle or 'Not determined'}")
        y_pos -= 20
        c.drawString(50, y_pos, f"Confidence Score: {confidence_score:.2f}")
        y_pos -= 40
        
        if results and 'collision_events' in results:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_pos, "Collision Events:")
            y_pos -= 20
            
            c.setFont("Helvetica", 10)
            for i, event in enumerate(results['collision_events']):
                c.drawString(70, y_pos, f"Event {i+1}: Frame {event['frame']}, Severity: {event['severity']}")
                y_pos -= 15
        
        c.save()
        
        return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        return jsonify({'error': 'Failed to generate report'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': device,
        'model_loaded': yolo_model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info(f"Starting Crash Analyzer API on device: {device}")
    app.run(debug=True, host='0.0.0.0', port=5000)
