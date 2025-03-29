import cv2
import numpy as np
import mediapipe as mp
import time

class EmotionDetector:
    """Emotion detector using MediaPipe Face Mesh"""
    
    def __init__(self):
        """Initialize the emotion detector with MediaPipe"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye blink detection
        self.eye_openness_history = []
        self.consecutive_low_eye_openness = 0
        
        # Frame tracking
        self.frame_count = 0
        self.last_detected_emotion = "detecting..."
        self.last_emotion_change_time = time.time()
        
        print("MediaPipe Face Mesh initialized for emotion detection")

    def detect_emotion(self, frame):
        """
        Detect emotion based on facial landmarks
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            dict: Emotion detection results with emotion, eye_openness, etc.
        """
        self.frame_count += 1
        
        try:
            # Convert to RGB (MediaPipe requires RGB input)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return {"emotion": "No face detected"}
            
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get eye landmarks
            # Left eye: landmarks 159, 145
            # Right eye: landmarks 386, 374
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]
            right_eye_top = face_landmarks.landmark[386]
            right_eye_bottom = face_landmarks.landmark[374]
            
            # Calculate eye openness
            left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
            right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
            eye_openness = (left_eye_height + right_eye_height) / 2
            
            # Get mouth landmarks
            # Mouth corners: landmarks 61, 291
            # Top and bottom: landmarks 0, 17
            mouth_left = face_landmarks.landmark[61]
            mouth_right = face_landmarks.landmark[291]
            mouth_top = face_landmarks.landmark[0]
            mouth_bottom = face_landmarks.landmark[17]
            
            # Check for smile - mouth width to height ratio
            mouth_width = abs(mouth_right.x - mouth_left.x)
            mouth_height = abs(mouth_top.y - mouth_bottom.y)
            smile_ratio = mouth_width / (mouth_height + 0.01)  # Avoid division by zero
            
            # Track eye openness
            self.eye_openness_history.append(eye_openness)
            if len(self.eye_openness_history) > 20:
                self.eye_openness_history = self.eye_openness_history[-20:]
            
            # Check if eyes are consistently less open
            TIREDNESS_THRESHOLD = 0.0145
            if eye_openness < TIREDNESS_THRESHOLD:
                self.consecutive_low_eye_openness += 1
            else:
                self.consecutive_low_eye_openness = 0
            
            if self.consecutive_low_eye_openness > 8:
                emotion = "tired"
            elif smile_ratio > 1.44:
                emotion = "happy"
            elif smile_ratio > 1.2:
                emotion = "neutral"
            elif smile_ratio < 1.2:
                emotion = "sad"
            else:
                emotion = "neutral"
            
            # Only log when emotion changes
            if emotion != self.last_detected_emotion:
                print(f"Emotion changed from {self.last_detected_emotion} to {emotion}")
                self.last_detected_emotion = emotion
                self.last_emotion_change_time = time.time()
            
            return {
                "emotion": emotion,
                "eye_openness": float(eye_openness),
                "consecutive_low": int(self.consecutive_low_eye_openness),
                "smile_ratio": float(smile_ratio),
                "success": True
            }
            
        except Exception as e:
            print(f"Error in emotion detection: {str(e)}")
            return {"error": str(e), "emotion": "detecting..."}