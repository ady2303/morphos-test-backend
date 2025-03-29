import asyncio
import base64
import cv2
import numpy as np
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import time
import math
import os
import dotenv
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from ultralytics import YOLO

# Import the emotion detector and Spotify service
from EmotionDetector import EmotionDetector
from SpotifyService import SpotifyService

# Load environment variables
dotenv.load_dotenv()

# Create FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
emotion_detector = EmotionDetector()
spotify_service = SpotifyService()

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

    async def send_json(self, data, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(data)

# Create connection manager
manager = ConnectionManager()

# Define the pose detector class
class TPoseDetector:
    def __init__(self):
        # Load YOLOv8 pose estimation model
        self.model = YOLO('yolov8n-pose.pt')
        
        # Exercise mode
        self.exercise_mode = "tpose"  # Can be "tpose" or "bicep_curl"
        
        # Bicep curl rep counting
        self.bicep_curl_state = "down"  # "down", "up", or "transitioning"
        self.rep_count = 0
        self.last_rep_time = 0
        self.primary_arm = "right"  # Default, will be detected automatically
        self.prev_angles = []  # For smoothing
        self.angle_threshold_up = 140  # Angle above which we consider the arm extended
        self.angle_threshold_down = 80  # Angle below which we consider the arm curled
        self.consecutive_frames_required = 3  # Frames required to confirm a state change
        self.frames_in_state = 0  # Counter for consecutive frames in current state
        
        # Keypoint indices for pose landmarks
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.NECK = 17  # Some models may not have this, will approximate if needed
        
        # Thresholds for pose detection
        self.ANGLE_THRESHOLD = 35  # Degrees of tolerance for arm angles
        self.CONFIDENCE_THRESHOLD = 0.25  # Lowered confidence threshold to be more lenient
        self.HORIZONTAL_TOLERANCE = 35  # Increased tolerance for horizontal alignment
        self.POSTURE_THRESHOLD = 12  # More lenient threshold for posture
        
        # Score history for smoother feedback
        self.score_history = []
        
    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def is_horizontal(self, point1, point2, tolerance=None):
        """Check if a line between two points is horizontal"""
        if tolerance is None:
            tolerance = self.HORIZONTAL_TOLERANCE
        return abs(point1[1] - point2[1]) < tolerance
    
    def check_t_pose(self, keypoints, confidences):
        """Check if the pose is a T-pose and provide detailed feedback."""
        feedback = {
            "left_arm": {"correct": False, "message": "", "score": 0},
            "right_arm": {"correct": False, "message": "", "score": 0},
            "posture": {"correct": False, "message": "", "score": 0},
            "overall": {"correct": False, "message": "", "score": 0}
        }
        
        # Check if all necessary keypoints are detected with sufficient confidence
        required_keypoints = [
            self.LEFT_SHOULDER, self.RIGHT_SHOULDER, 
            self.LEFT_ELBOW, self.RIGHT_ELBOW,
            self.LEFT_WRIST, self.RIGHT_WRIST,
            self.LEFT_HIP, self.RIGHT_HIP
        ]
        
        for kp in required_keypoints:
            if kp >= len(confidences) or confidences[kp] < self.CONFIDENCE_THRESHOLD:
                feedback["overall"]["message"] = "Move further"
                return False, feedback
        
        # Get coordinates for key joints
        left_shoulder = keypoints[self.LEFT_SHOULDER]
        right_shoulder = keypoints[self.RIGHT_SHOULDER]
        left_elbow = keypoints[self.LEFT_ELBOW]
        right_elbow = keypoints[self.RIGHT_ELBOW]
        left_wrist = keypoints[self.LEFT_WRIST]
        right_wrist = keypoints[self.RIGHT_WRIST]
        left_hip = keypoints[self.LEFT_HIP]
        right_hip = keypoints[self.RIGHT_HIP]
        
        # Calculate midpoints for reference
        shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                        (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_mid = ((left_hip[0] + right_hip[0]) / 2, 
                   (left_hip[1] + right_hip[1]) / 2)
        
        # Check left arm (straight and horizontal)
        left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        left_arm_straightness = min(100, max(0, (left_arm_angle - (180 - self.ANGLE_THRESHOLD)) / self.ANGLE_THRESHOLD * 100))
        is_left_arm_straight = left_arm_angle > (180 - self.ANGLE_THRESHOLD)
        
        left_arm_horizontal_diff = abs(left_shoulder[1] - left_wrist[1])
        left_arm_horizontalness = min(100, max(0, (self.HORIZONTAL_TOLERANCE - left_arm_horizontal_diff) / self.HORIZONTAL_TOLERANCE * 100))
        is_left_arm_horizontal = left_arm_horizontal_diff < self.HORIZONTAL_TOLERANCE
        
        feedback["left_arm"]["score"] = (left_arm_straightness + left_arm_horizontalness) / 2
        
        if is_left_arm_straight and is_left_arm_horizontal:
            feedback["left_arm"]["correct"] = True
            feedback["left_arm"]["message"] = "Left arm position is good!"
        else:
            if not is_left_arm_straight:
                feedback["left_arm"]["message"] = "Straighten your left arm"
            elif not is_left_arm_horizontal:
                feedback["left_arm"]["message"] = "Level your left arm"
        
        # Check right arm (straight and horizontal)
        right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        right_arm_straightness = min(100, max(0, (right_arm_angle - (180 - self.ANGLE_THRESHOLD)) / self.ANGLE_THRESHOLD * 100))
        is_right_arm_straight = right_arm_angle > (180 - self.ANGLE_THRESHOLD)
        
        right_arm_horizontal_diff = abs(right_shoulder[1] - right_wrist[1])
        right_arm_horizontalness = min(100, max(0, (self.HORIZONTAL_TOLERANCE - right_arm_horizontal_diff) / self.HORIZONTAL_TOLERANCE * 100))
        is_right_arm_horizontal = right_arm_horizontal_diff < self.HORIZONTAL_TOLERANCE
        
        feedback["right_arm"]["score"] = (right_arm_straightness + right_arm_horizontalness) / 2
        
        if is_right_arm_straight and is_right_arm_horizontal:
            feedback["right_arm"]["correct"] = True
            feedback["right_arm"]["message"] = "Right arm position is good!"
        else:
            if not is_right_arm_straight:
                feedback["right_arm"]["message"] = "Straighten your right arm"
            elif not is_right_arm_horizontal:
                feedback["right_arm"]["message"] = "Level your right arm"
        
        # Check posture
        spine_angle = self.calculate_angle(shoulder_mid, hip_mid, (hip_mid[0], hip_mid[1] + 10))
        deviation_from_upright = abs(180 - spine_angle)
        posture_score = min(100, max(0, (self.POSTURE_THRESHOLD - deviation_from_upright) / self.POSTURE_THRESHOLD * 100))
        is_upright = deviation_from_upright < self.POSTURE_THRESHOLD
        
        feedback["posture"]["score"] = posture_score
        
        if is_upright:
            feedback["posture"]["correct"] = True
            feedback["posture"]["message"] = "Posture is perfect!"
        else:
            feedback["posture"]["message"] = "Stand more upright"
        
        # Calculate overall score
        overall_score = (feedback["left_arm"]["score"] + feedback["right_arm"]["score"] + feedback["posture"]["score"]) / 3
        feedback["overall"]["score"] = overall_score
        
        # Overall T-pose check
        is_t_pose = overall_score >= 75  # Need at least 75% overall score
        
        if is_t_pose:
            feedback["overall"]["correct"] = True
            feedback["overall"]["message"] = "PERFECT T-POSE!"
        else:
            if overall_score > 60:
                feedback["overall"]["message"] = "Almost there!"
            elif overall_score > 40:
                feedback["overall"]["message"] = "Getting better!"
            else:
                feedback["overall"]["message"] = "Keep trying!"
        
        return is_t_pose, feedback
    
    def check_bicep_curl_pose(self, keypoints, confidences):
        """
        Check the bicep curl position and count repetitions dynamically.
        
        Args:
            keypoints: Array of detected keypoints
            confidences: Confidence values for each keypoint
        
        Returns:
            is_correct: Boolean indicating if the pose is correct
            feedback: Dictionary with feedback on each part of the pose
        """
        feedback = {
            "arm_position": {"correct": False, "message": "", "score": 0},
            "elbow_angle": {"correct": False, "message": "", "score": 0},
            "posture": {"correct": False, "message": "", "score": 0},
            "overall": {"correct": False, "message": "", "score": 0},
            "rep_phase": {"status": self.bicep_curl_state, "count": self.rep_count}
        }
        
        # Check if all necessary keypoints are detected with sufficient confidence
        required_keypoints = [
            self.LEFT_SHOULDER, self.RIGHT_SHOULDER, 
            self.LEFT_ELBOW, self.RIGHT_ELBOW,
            self.LEFT_WRIST, self.RIGHT_WRIST,
            self.LEFT_HIP, self.RIGHT_HIP
        ]
        
        for kp in required_keypoints:
            if kp >= len(confidences) or confidences[kp] < self.CONFIDENCE_THRESHOLD:
                feedback["overall"]["message"] = "Move closer"
                return False, feedback
        
        # Get coordinates for key joints
        left_shoulder = keypoints[self.LEFT_SHOULDER]
        right_shoulder = keypoints[self.RIGHT_SHOULDER]
        left_elbow = keypoints[self.LEFT_ELBOW]
        right_elbow = keypoints[self.RIGHT_ELBOW]
        left_wrist = keypoints[self.LEFT_WRIST]
        right_wrist = keypoints[self.RIGHT_WRIST]
        left_hip = keypoints[self.LEFT_HIP]
        right_hip = keypoints[self.RIGHT_HIP]
        
        # Calculate midpoints for reference
        shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                        (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_mid = ((left_hip[0] + right_hip[0]) / 2, 
                   (left_hip[1] + right_hip[1]) / 2)
        
        # Calculate arm angles for bicep curl tracking
        left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Determine which arm is doing the curl (the one with the smaller angle is more likely curling)
        # We also want to maintain some consistency between frames, so we don't switch too often
        left_curling_score = min(180, max(0, 180 - left_arm_angle))
        right_curling_score = min(180, max(0, 180 - right_arm_angle))
        
        # If the scores are similar, stick with the previous arm to avoid flickering
        if abs(left_curling_score - right_curling_score) < 20:
            primary_side = self.primary_arm
        else:
            self.primary_arm = "left" if left_curling_score > right_curling_score else "right"
            primary_side = self.primary_arm
        
        # Get the current angle for the primary arm
        current_angle = left_arm_angle if primary_side == "left" else right_arm_angle
        
        # Apply some smoothing to the angle
        self.prev_angles.append(current_angle)
        if len(self.prev_angles) > 5:  # Keep a rolling window of 5 frames
            self.prev_angles.pop(0)
        smoothed_angle = sum(self.prev_angles) / len(self.prev_angles)
        
        # Track bicep curl rep state based on angle
        # Determine the current position of the curl
        prev_state = self.bicep_curl_state
        
        # Check if we're in the "up" position (arm is extended)
        if smoothed_angle > self.angle_threshold_up:
            if self.bicep_curl_state != "up":
                self.frames_in_state += 1
                if self.frames_in_state >= self.consecutive_frames_required:
                    # If we were previously "down", count a rep when transitioning to "up"
                    if self.bicep_curl_state == "down":
                        current_time = time.time()
                        # Ensure we haven't just counted a rep (to prevent double counting)
                        if current_time - self.last_rep_time > 0.5:
                            self.rep_count += 1
                            self.last_rep_time = current_time
                    self.bicep_curl_state = "up"
                    self.frames_in_state = 0
            else:
                self.frames_in_state = 0
        # Check if we're in the "down" position (arm is curled)
        elif smoothed_angle < self.angle_threshold_down:
            if self.bicep_curl_state != "down":
                self.frames_in_state += 1
                if self.frames_in_state >= self.consecutive_frames_required:
                    self.bicep_curl_state = "down"
                    self.frames_in_state = 0
            else:
                self.frames_in_state = 0
        # We're in a transitioning state
        else:
            if self.bicep_curl_state != "transitioning":
                self.frames_in_state += 1
                if self.frames_in_state >= self.consecutive_frames_required:
                    self.bicep_curl_state = "transitioning"
                    self.frames_in_state = 0
            else:
                self.frames_in_state = 0
                
        # Update the feedback with the current rep state
        feedback["rep_phase"]["status"] = self.bicep_curl_state
        feedback["rep_phase"]["count"] = self.rep_count
        
        # Based on the current angle, determine if we're in a proper curl position
        # We're more lenient now - any position during a curl is fine
        elbow_score = 0
        
        if self.bicep_curl_state == "down":
            # For down position, score is better the closer to the threshold_down or below
            elbow_score = 100 - min(100, max(0, (smoothed_angle - self.angle_threshold_down) * 3))
            if smoothed_angle <= self.angle_threshold_down + 15:  # More lenient
                is_elbow_good = True
                feedback["elbow_angle"]["message"] = "Good curl position"
            else:
                is_elbow_good = False
                feedback["elbow_angle"]["message"] = "Curl higher"
        elif self.bicep_curl_state == "up":
            # For up position, score is better the closer to the threshold_up or above
            elbow_score = 100 - min(100, max(0, (self.angle_threshold_up - smoothed_angle) * 3))
            if smoothed_angle >= self.angle_threshold_up - 15:  # More lenient
                is_elbow_good = True
                feedback["elbow_angle"]["message"] = "Good extension"
            else:
                is_elbow_good = False
                feedback["elbow_angle"]["message"] = "Extend fully"
        else:  # transitioning
            # For transition, any angle between thresholds is good
            range_size = self.angle_threshold_up - self.angle_threshold_down
            position_in_range = (smoothed_angle - self.angle_threshold_down) / range_size
            elbow_score = 100 * position_in_range  # Score based on transition progress
            is_elbow_good = True
            feedback["elbow_angle"]["message"] = "Good transition"
        
        feedback["elbow_angle"]["score"] = elbow_score
        feedback["elbow_angle"]["correct"] = is_elbow_good
        
        # For dynamic bicep curls, we're more lenient with arm position
        # We just want to ensure the arm isn't too far from the body
        if primary_side == "left":
            shoulder_hip_dist = np.sqrt((left_shoulder[0] - left_hip[0])**2 + (left_shoulder[1] - left_hip[1])**2)
            shoulder_elbow_dist = np.sqrt((left_shoulder[0] - left_elbow[0])**2)
            arm_position_score = 100 - min(100, (shoulder_elbow_dist / shoulder_hip_dist) * 100)
            is_arm_close = shoulder_elbow_dist < shoulder_hip_dist * 0.4  # More lenient: 0.3 -> 0.4
        else:
            shoulder_hip_dist = np.sqrt((right_shoulder[0] - right_hip[0])**2 + (right_shoulder[1] - right_hip[1])**2)
            shoulder_elbow_dist = np.sqrt((right_shoulder[0] - right_elbow[0])**2)
            arm_position_score = 100 - min(100, (shoulder_elbow_dist / shoulder_hip_dist) * 100)
            is_arm_close = shoulder_elbow_dist < shoulder_hip_dist * 0.4  # More lenient: 0.3 -> 0.4
        
        feedback["arm_position"]["score"] = arm_position_score
        
        if is_arm_close:
            feedback["arm_position"]["correct"] = True
            feedback["arm_position"]["message"] = "Arm position good"
        else:
            feedback["arm_position"]["message"] = "Keep arm closer"
        
        # Check posture (same as before but more lenient)
        spine_angle = self.calculate_angle(shoulder_mid, hip_mid, (hip_mid[0], hip_mid[1] + 10))
        deviation_from_upright = abs(180 - spine_angle)
        posture_score = min(100, max(0, (self.POSTURE_THRESHOLD - deviation_from_upright) / self.POSTURE_THRESHOLD * 100))
        is_upright = deviation_from_upright < self.POSTURE_THRESHOLD
        
        feedback["posture"]["score"] = posture_score
        
        if is_upright:
            feedback["posture"]["correct"] = True
            feedback["posture"]["message"] = "Posture is good"
        else:
            feedback["posture"]["message"] = "Stand upright"
        
        # For dynamic exercises, we focus more on the motion than perfect form
        # Emphasize the elbow angle for scoring
        overall_score = (feedback["elbow_angle"]["score"] * 2 + 
                        feedback["arm_position"]["score"] + 
                        feedback["posture"]["score"]) / 4
        feedback["overall"]["score"] = overall_score
        
        # Lower the threshold for a "correct" pose
        is_correct_pose = overall_score >= 65  # More lenient: 75 -> 65
        
        # Adapt the feedback message based on rep counts and current state
        if is_correct_pose:
            feedback["overall"]["correct"] = True
            if self.bicep_curl_state == "up":
                feedback["overall"]["message"] = "Now curl down"
            elif self.bicep_curl_state == "down":
                feedback["overall"]["message"] = "Now extend up"
            else:
                feedback["overall"]["message"] = f"Rep: {self.rep_count}"
        else:
            if overall_score > 50:
                if self.bicep_curl_state == "up":
                    feedback["overall"]["message"] = "Extend fully"
                elif self.bicep_curl_state == "down":
                    feedback["overall"]["message"] = "Curl higher"
                else:
                    feedback["overall"]["message"] = "Keep going!"
            elif overall_score > 35:
                feedback["overall"]["message"] = "Getting better!"
            else:
                feedback["overall"]["message"] = "Try again"
        
        return is_correct_pose, feedback
    
    def process_frame(self, frame):
        """Process a frame and return pose data and feedback."""
        try:
            # Run detection
            results = self.model(frame, verbose=False)
            
            # Prepare default feedback based on current mode
            if self.exercise_mode == "tpose":
                default_feedback = {
                    "left_arm": {"correct": False, "message": "Not visible", "score": 0},
                    "right_arm": {"correct": False, "message": "Not visible", "score": 0},
                    "posture": {"correct": False, "message": "Not visible", "score": 0},
                    "overall": {"correct": False, "message": "No detection", "score": 0}
                }
            else:  # bicep_curl mode
                default_feedback = {
                    "elbow_angle": {"correct": False, "message": "Not visible", "score": 0},
                    "arm_position": {"correct": False, "message": "Not visible", "score": 0},
                    "posture": {"correct": False, "message": "Not visible", "score": 0},
                    "overall": {"correct": False, "message": "No detection", "score": 0},
                    "rep_phase": {"status": self.bicep_curl_state, "count": self.rep_count}
                }
            
            # Check if any person is detected
            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                # Get the first person detected (assume only one person in frame)
                keypoints = results[0].keypoints.data[0].cpu().numpy()[:, :2]  # Get x, y coordinates
                confidences = results[0].keypoints.data[0].cpu().numpy()[:, 2]  # Get confidence scores
                
                # Check pose based on current mode
                if self.exercise_mode == "tpose":
                    is_correct_pose, feedback = self.check_t_pose(keypoints, confidences)
                else:  # bicep_curl mode
                    is_correct_pose, feedback = self.check_bicep_curl_pose(keypoints, confidences)
                
                # Convert keypoints to list for JSON serialization
                keypoints_list = keypoints.tolist()
                
                return {
                    "keypoints": keypoints_list,
                    "feedback": feedback,
                    "exercise_mode": self.exercise_mode
                }
            else:
                # No person detected
                return {
                    "keypoints": [],
                    "feedback": default_feedback,
                    "exercise_mode": self.exercise_mode
                }
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            # Return empty data on error
            return {
                "keypoints": [],
                "feedback": default_feedback,
                "exercise_mode": self.exercise_mode,
                "error": str(e)
            }

    def set_exercise_mode(self, mode):
        """Set the exercise mode."""
        if mode in ["tpose", "bicep_curl"]:
            self.exercise_mode = mode
            self.score_history = []  # Reset score history for new exercise
            if mode == "bicep_curl":
                self.rep_count = 0  # Reset rep counter
            return True
        return False

# Store client-specific detectors and emotion history
detectors = {}
emotion_histories = {}  # Store emotion history for each client
current_recommendations = {}  # Store current recommendations for each client

# Pydantic model for track recommendations
class Track(BaseModel):
    id: str
    title: str
    artist: str
    album: str
    image_url: Optional[str] = None
    preview_url: Optional[str] = None
    external_url: str

class TrackList(BaseModel):
    tracks: List[Track]

# Helper function to determine stable emotion
def determine_stable_emotion(emotion_history, threshold=0.6, min_samples=5):
    """
    Determine stable emotion from history
    
    Args:
        emotion_history: List of detected emotions
        threshold: Minimum percentage for an emotion to be considered stable
        min_samples: Minimum number of samples required
        
    Returns:
        str: The stable emotion or None
    """
    if len(emotion_history) < min_samples:
        return None
    
    # Count occurrences of each emotion
    counts = {}
    for emotion in emotion_history:
        if emotion != "detecting..." and emotion != "No face detected":
            counts[emotion] = counts.get(emotion, 0) + 1
    
    # Find the most frequent emotion
    max_count = 0
    dominant = None
    
    for emotion, count in counts.items():
        if count > max_count:
            max_count = count
            dominant = emotion
    
    # Check if dominant emotion appears enough times
    if dominant and max_count / len(emotion_history) >= threshold:
        return dominant
    
    return None

# Route to get recommendations based on emotion
@app.get("/api/recommendations", response_model=TrackList)
async def get_recommendations(emotion: str = Query(...), limit: int = Query(3, ge=1, le=10)):
    """Get music recommendations based on emotion"""
    if emotion not in spotify_service.emotion_to_music:
        raise HTTPException(status_code=400, detail=f"Unknown emotion: {emotion}")
        
    recommendations = spotify_service.get_recommendations_for_emotion(emotion, limit)
    return recommendations

# Route to get available Spotify genres
@app.get("/api/genres")
async def get_genres():
    """Get a list of available genre seeds from Spotify"""
    genres = spotify_service.get_available_genres()
    return genres

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    # Create a new detector for this client if it doesn't exist
    if client_id not in detectors:
        print(f"Creating new detector for client {client_id}")
        detectors[client_id] = TPoseDetector()
        emotion_histories[client_id] = []
    
    detector = detectors[client_id]
    
    await manager.connect(websocket, client_id)
    print(f"Client {client_id} connected")
    
    try:
        while True:
            # Receive the frame data from the client
            data = await websocket.receive_text()
            
            # Extract the image data and command
            try:
                json_data = json.loads(data)
                
                # Check if this is a command
                if "command" in json_data:
                    command = json_data["command"]
                    print(f"Received command: {command}")
                    
                    if command == "set_mode":
                        mode = json_data.get("mode", "tpose")
                        success = detector.set_exercise_mode(mode)
                        print(f"Set mode to {mode}, success: {success}")
                        await manager.send_json({"type": "mode_change", "mode": detector.exercise_mode, "success": success}, client_id)
                        continue
                
                    # Handle explicit request for music recommendations
                    if command == "get_recommendations":
                        emotion = json_data.get("emotion", "neutral")
                        limit = json_data.get("limit", 3)
                        
                        # Get recommendations
                        recommendations = spotify_service.get_recommendations_for_emotion(emotion, limit)
                        current_recommendations[client_id] = recommendations
                        
                        # Send recommendations to client
                        await manager.send_json({
                            "type": "music_recommendations", 
                            "data": recommendations
                        }, client_id)
                        continue
                
                # Process image frame
                if "image" in json_data:
                    # Decode the base64 image
                    img_data = base64.b64decode(json_data["image"].split(',')[1])
                    
                    # Convert to numpy array
                    np_arr = np.frombuffer(img_data, np.uint8)
                    
                    # Decode the image
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        print("WARNING: Frame is None after decoding")
                        continue
                        
                    # Process the frame for pose detection
                    pose_result = detector.process_frame(frame)
                    
                    # Process the frame for emotion detection
                    emotion_result = emotion_detector.detect_emotion(frame)
                    
                    # Update emotion history
                    if "emotion" in emotion_result and emotion_result["emotion"] not in ["detecting...", "No face detected"]:
                        emotion_histories[client_id].append(emotion_result["emotion"])
                        # Keep only the last 20 emotions
                        if len(emotion_histories[client_id]) > 20:
                            emotion_histories[client_id] = emotion_histories[client_id][-20:]
                    
                    # Determine stable emotion
                    stable_emotion = determine_stable_emotion(emotion_histories[client_id])
                    
                    # Check if we need to get new recommendations
                    if stable_emotion and (
                        client_id not in current_recommendations or 
                        len(current_recommendations[client_id].get("tracks", [])) == 0
                    ):
                        # Get music recommendations
                        recommendations = spotify_service.get_recommendations_for_emotion(stable_emotion, 3)
                        current_recommendations[client_id] = recommendations
                    else:
                        # Use existing recommendations
                        recommendations = current_recommendations.get(client_id, {"tracks": []})
                    
                    # Combine results and send to client
                    combined_result = {
                        "type": "combined_data",
                        "pose_data": pose_result,
                        "emotion_data": {
                            "current_emotion": emotion_result.get("emotion", "unknown"),
                            "stable_emotion": stable_emotion,
                            "eye_openness": emotion_result.get("eye_openness", 0),
                        },
                        "music_recommendations": recommendations
                    }
                    
                    # Send the result back to the client
                    await manager.send_json(combined_result, client_id)
                    
            except json.JSONDecodeError:
                print("Invalid JSON data received")
                await manager.send_json({"type": "error", "message": "Invalid JSON format"}, client_id)
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                await manager.send_json({"type": "error", "message": str(e)}, client_id)
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        print(f"Client {client_id} disconnected")

# Root endpoint for health check
@app.get("/")
async def root():
    return {
        "status": "Fitness Coach and Emotion Detection API is running",
        "spotify_available": spotify_service.sp is not None
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    print(f"Spotify API integration: {'Available' if spotify_service.sp is not None else 'Not available - check your credentials'}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)