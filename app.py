import cv2
import dlib
import numpy as np
import pymongo
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance

class AdvancedFocusTracker:
    def __init__(self, mongo_uri='mongodb://localhost:27017/', db_name='focus_tracker'):
        # MongoDB initialization
        try:
            self.client = pymongo.MongoClient(mongo_uri)
            self.db = self.client[db_name]
            self.sessions_collection = self.db['focus_sessions']
        except Exception as e:
            print(f"MongoDB connection error: {e}")
            self.client = None
        
        # Facial recognition setup
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        # Focus tracking variables
        self.total_focus_time = timedelta()
        self.focus_start_time = None
        self.is_focused = False
        
        # Distraction tracking
        self.distraction_log = []
        self.max_consecutive_distractions = 0
        self.current_consecutive_distractions = 0
        
        # Calibration
        self.baseline_ear = None
    
    def calibrate_eye_tracking(self, num_frames=100):
        """Calibrate baseline eye aspect ratio"""
        print("Starting eye tracking calibration...")
        cap = cv2.VideoCapture(1)
        ear_values = []
        
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            if faces:
                face = faces[0]
                landmarks = self.predictor(gray, face)
                
                # Get eye coordinates
                left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
                right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
                
                # Calculate eye aspect ratios
                left_ear = self.get_eye_aspect_ratio(left_eye)
                right_ear = self.get_eye_aspect_ratio(right_eye)
                
                ear_values.extend([left_ear, right_ear])
        
        cap.release()
        
        # Calculate baseline
        if ear_values:
            self.baseline_ear = np.mean(ear_values)
            print(f"Calibration complete. Baseline EAR: {self.baseline_ear}")
        else:
            print("Calibration failed. Unable to detect face.")
        
        return self.baseline_ear
    
    def get_eye_aspect_ratio(self, eye_landmarks):
        """Calculate Eye Aspect Ratio for detecting eye openness"""
        # Vertical eye distances
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        # Horizontal eye distance
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def check_focus(self, frame):
        """Comprehensive focus checking method"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            self.update_focus_state(False)
            return frame, "No Face Detected", False
        
        face = faces[0]  # Primary face
        landmarks = self.predictor(gray, face)
        
        # Get eye coordinates
        left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
        
        # Calculate eye aspect ratios
        left_ear = self.get_eye_aspect_ratio(left_eye)
        right_ear = self.get_eye_aspect_ratio(right_eye)
        
        # Use baseline for more accurate detection if calibrated
        if self.baseline_ear is None:
            self.calibrate_eye_tracking()
        
        # Adaptive focus criteria
        baseline = self.baseline_ear if self.baseline_ear else 0.2
        is_looking = (
            left_ear > baseline * 0.7 and  # More robust detection
            right_ear > baseline * 0.7 and 
            left_ear < baseline * 1.5 and  # Prevent false positives
            right_ear < baseline * 1.5
        )
        
        self.update_focus_state(is_looking)
        
        # Visualization
        if is_looking:
            color = (0, 255, 0)  # Green for focused
            status = "Focused"
        else:
            color = (0, 0, 255)  # Red for not focused
            status = "Distracted"
            self.distraction_log.append(datetime.now())
        
        # Draw rectangle around face
        cv2.rectangle(frame, 
                      (face.left(), face.top()), 
                      (face.right(), face.bottom()), 
                      color, 2)
        
        return frame, status, is_looking
    
    def update_focus_state(self, is_looking):
        """Update focus tracking state"""
        current_time = datetime.now()
        
        if is_looking and not self.is_focused:
            self.focus_start_time = current_time
            self.is_focused = True
            self.current_consecutive_distractions = 0
        
        elif not is_looking and self.is_focused:
            if self.focus_start_time:
                self.total_focus_time += current_time - self.focus_start_time
                self.current_consecutive_distractions += 1
                self.max_consecutive_distractions = max(
                    self.max_consecutive_distractions, 
                    self.current_consecutive_distractions
                )
            self.is_focused = False
    
    def generate_focus_report(self):
        """Generate a comprehensive focus report"""
        total_session_time = datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        focus_percentage = (self.total_focus_time.total_seconds() / total_session_time.total_seconds()) * 100
        
        # Prepare session document
        session_doc = {
            'date': datetime.now(),
            'total_time': self.total_focus_time.total_seconds(),
            'focus_percentage': focus_percentage,
            'total_distractions': len(self.distraction_log),
            'max_consecutive_distractions': self.max_consecutive_distractions,
            'distraction_timestamps': self.distraction_log
        }
        
        # Save to MongoDB if connection exists
        if self.client:
            try:
                self.sessions_collection.insert_one(session_doc)
            except Exception as e:
                print(f"Failed to save session to MongoDB: {e}")
        
        return {
            'total_focus_time': self.total_focus_time,
            'focus_percentage': focus_percentage,
            'total_distractions': len(self.distraction_log),
            'max_consecutive_distractions': self.max_consecutive_distractions
        }

def main():
    # Initialize tracker
    tracker = AdvancedFocusTracker()
    
    # Calibrate eye tracking
    tracker.calibrate_eye_tracking()
    
    # Open video capture
    cap = cv2.VideoCapture(1)  # Use 0 for default camera, or change to 1/2 if needed
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process frame
        frame, status, is_focused = tracker.check_focus(frame)
        
        # Calculate focus time
        focus_time = tracker.total_focus_time
        
        # Display information on frame
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 255, 0) if is_focused else (0, 0, 255), 2)
        cv2.putText(frame, f"Focus Time: {str(focus_time).split('.')[0]}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow("Advanced Focus Tracker", frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Generate final report
            report = tracker.generate_focus_report()
            print("Focus Report:", report)
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()