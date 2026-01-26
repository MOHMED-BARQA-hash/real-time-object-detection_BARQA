#!/usr/bin/env python3
"""
Real-time Object Detection System
Author: Alex Chen
Created: 2024
Description: A professional real-time object detection system using YOLO and OpenCV
License: MIT
"""

import cv2
import numpy as np
import argparse
import time
import os
import sys
from typing import List, Tuple, Optional, Dict
import urllib.request
import subprocess

class ObjectDetector:
    """Main object detection class using YOLO model"""
    
    def __init__(self, model_type='yolov3-tiny', confidence=0.5, nms_threshold=0.4, use_gpu=False):
        self.confidence_threshold = confidence
        self.nms_threshold = nms_threshold
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.net = None
        self.classes = []
        self.colors = []
        self.output_layers = []
        self.is_initialized = False
        
    def setup_model(self):
        """Download and setup YOLO model"""
        print("Setting up object detection model...")
        
        # Create models directory if needed
        os.makedirs('models', exist_ok=True)
        
        # Download model files if they don't exist
        self._download_model_files()
        
        # Load model paths
        model_path, config_path, classes_path = self._get_model_paths()
        
        try:
            # Load YOLO network
            self.net = cv2.dnn.readNet(model_path, config_path)
            
            # Configure backend
            if self.use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("Using GPU acceleration")
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("Using CPU")
            
            # Load class names
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Get output layers
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Generate colors for classes
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            
            self.is_initialized = True
            print(f"Model loaded successfully: {self.model_type}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def _download_model_files(self):
        """Download required model files"""
        base_url = "https://github.com/pjreddie/darknet/raw/master/"
        files_to_download = {
            'yolov3-tiny.weights': f'{base_url}cfg/yolov3-tiny.weights?raw=true',
            'yolov3-tiny.cfg': f'{base_url}cfg/yolov3-tiny.cfg',
            'coco.names': f'{base_url}data/coco.names'
        }
        
        for filename, url in files_to_download.items():
            filepath = os.path.join('models', filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, filepath)
                    print(f"Downloaded {filename}")
                except Exception as e:
                    print(f"Failed to download {filename}: {e}")
    
    def _get_model_paths(self) -> Tuple[str, str, str]:
        """Get paths to model files"""
        model_files = {
            'yolov3-tiny': ('yolov3-tiny.weights', 'yolov3-tiny.cfg'),
            'yolov3': ('yolov3.weights', 'yolov3.cfg')
        }
        
        if self.model_type not in model_files:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        weights_file, config_file = model_files[self.model_type]
        weights_path = os.path.join('models', weights_file)
        config_path = os.path.join('models', config_file)
        classes_path = os.path.join('models', 'coco.names')
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
        return weights_path, config_path, classes_path
    
    def detect_objects(self, frame):
        """Detect objects in frame"""
        if not self.is_initialized:
            self.setup_model()
        
        height, width = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416), (0, 0, 0), 
            True, crop=False
        )
        
        # Run inference
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process outputs
        boxes, confidences, class_ids = self._process_outputs(outputs, width, height)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, 
            self.confidence_threshold, 
            self.nms_threshold
        )
        
        return boxes, confidences, class_ids, indices
    
    def _process_outputs(self, outputs, width, height):
        """Process YOLO outputs"""
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Convert center coordinates to box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        return boxes, confidences, class_ids
    
    def draw_detections(self, frame, boxes, confidences, class_ids, indices):
        """Draw detection boxes on frame"""
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = self.colors[class_ids[i]]
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw label background
                label_text = f'{label}: {confidence:.2f}'
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(
                    frame, 
                    (x, y - label_size[1] - 10), 
                    (x + label_size[0], y), 
                    color, 
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame, label_text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )
        
        return frame

class DetectionVisualizer:
    """Handles visualization of detection results"""
    
    def __init__(self):
        self.detection_history = []
        self.max_history = 30
    
    def add_analytics(self, frame, num_detections):
        """Add analytics information to frame"""
        height = frame.shape[0]
        
        # Add object count
        count_text = f'Objects: {num_detections}'
        cv2.putText(
            frame, count_text, (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        
        return frame
    
    def draw_fps(self, frame, fps):
        """Draw FPS counter on frame"""
        fps_text = f'FPS: {fps:.1f}'
        cv2.putText(
            frame, fps_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        return frame

class RealTimeDetectionApp:
    """Main application class"""
    
    def __init__(self):
        self.detector = None
        self.visualizer = DetectionVisualizer()
        self.running = False
        
    def setup_detector(self, model_type, confidence, nms_threshold, use_gpu):
        """Initialize the object detector"""
        self.detector = ObjectDetector(
            model_type=model_type,
            confidence=confidence,
            nms_threshold=nms_threshold,
            use_gpu=use_gpu
        )
    
    def run_webcam_detection(self, camera_id=0, output_file=None):
        """Run object detection on webcam feed"""
        if not self.detector:
            print("Detector not initialized!")
            return
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Setup video writer if output specified
        writer = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
        
        print("\n" + "="*50)
        print("Real-time Object Detection Started!")
        print("="*50)
        print("Controls:")
        print("  Q - Quit application")
        print("  S - Save screenshot")
        print("  P - Pause/resume detection")
        print("  + - Increase confidence threshold")
        print("  - - Decrease confidence threshold")
        print("="*50)
        
        self.running = True
        prev_time = 0
        screenshot_count = 0
        paused = False
        
        try:
            while self.running:
                if not paused:
                    # Read frame
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Failed to capture frame")
                        break
                    
                    # Detect objects
                    boxes, confidences, class_ids, indices = self.detector.detect_objects(frame)
                    
                    # Draw detections
                    frame = self.detector.draw_detections(frame, boxes, confidences, class_ids, indices)
                    
                    # Add analytics
                    num_detections = len(indices) if indices else 0
                    frame = self.visualizer.add_analytics(frame, num_detections)
                    
                    # Calculate and display FPS
                    current_time = time.time()
                    fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 0
                    prev_time = current_time
                    frame = self.visualizer.draw_fps(frame, fps)
                    
                    # Write to output file
                    if writer:
                        writer.write(frame)
                
                # Display frame
                cv2.imshow('Real-time Object Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_count = self._save_screenshot(frame, screenshot_count)
                elif key == ord('p'):
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"Detection {status}")
                elif key == ord('+'):
                    new_conf = min(0.95, self.detector.confidence_threshold + 0.05)
                    self.detector.confidence_threshold = new_conf
                    print(f"Confidence threshold increased to: {new_conf:.2f}")
                elif key == ord('-'):
                    new_conf = max(0.1, self.detector.confidence_threshold - 0.05)
                    self.detector.confidence_threshold = new_conf
                    print(f"Confidence threshold decreased to: {new_conf:.2f}")
        
        except KeyboardInterrupt:
            print("\nDetection interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.running = False
            print("Detection session ended")
    
    def _save_screenshot(self, frame, count):
        """Save current frame as screenshot"""
        count += 1
        filename = f'detection_screenshot_{count:03d}.jpg'
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
        return count
    
    def process_video_file(self, video_path, output_path=None):
        """Process video file for object detection"""
        if not self.detector:
            print("Detector not initialized!")
            return
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"FPS: {fps:.1f}")
        print(f"Total frames: {total_frames}")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            print(f"Output will be saved to: {output_path}")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            boxes, confidences, class_ids, indices = self.detector.detect_objects(frame)
            frame = self.detector.draw_detections(frame, boxes, confidences, class_ids, indices)
            
            # Add info overlay
            num_detections = len(indices) if indices else 0
            frame = self.visualizer.add_analytics(frame, num_detections)
            
            # Progress info
            progress = (frame_count / total_frames) * 100
            progress_text = f'Progress: {progress:.1f}%'
            cv2.putText(frame, progress_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if writer:
                writer.write(frame)
            
            # Show progress every 10 frames
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps_processed = frame_count / elapsed if elapsed > 0 else 0
                print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%) - FPS: {fps_processed:.1f}")
            
            frame_count += 1
        
        cap.release()
        if writer:
            writer.release()
        
        total_time = time.time() - start_time
        print(f"Video processing completed in {total_time:.2f} seconds")
        print(f"Average FPS: {frame_count/total_time:.2f}")

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import cv2
        import numpy as np
        print("✓ OpenCV and NumPy are available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install opencv-python numpy")
        return False

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                   REAL-TIME OBJECT DETECTION                 ║
    ║                     Professional Edition                     ║
    ║                         Version 1.0                          ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Real-time Object Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Webcam detection with default settings
  python real_time_object_detection.py --source webcam

  # Webcam with custom model and confidence
  python real_time_object_detection.py --source webcam --model yolov3 --confidence 0.7

  # Process video file
  python real_time_object_detection.py --source video.mp4 --output output.avi

  # Use GPU acceleration
  python real_time_object_detection.py --source webcam --gpu

  # Different camera device
  python real_time_object_detection.py --source webcam --camera-id 1
        '''
    )
    
    parser.add_argument('--source', type=str, required=True,
                       help='Input source (webcam, or path to video file)')
    parser.add_argument('--model', type=str, default='yolov3-tiny',
                       choices=['yolov3-tiny', 'yolov3'],
                       help='YOLO model type (default: yolov3-tiny)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold 0.1-0.9 (default: 0.5)')
    parser.add_argument('--nms-threshold', type=float, default=0.4,
                       help='Non-maximum suppression threshold (default: 0.4)')
    parser.add_argument('--output', type=str,
                       help='Output file path for processed video')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration if available')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera device ID (default: 0)')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create and configure application
    app = RealTimeDetectionApp()
    app.setup_detector(
        model_type=args.model,
        confidence=args.confidence,
        nms_threshold=args.nms_threshold,
        use_gpu=args.gpu
    )
    
    # Run detection based on source type
    if args.source.lower() == 'webcam':
        app.run_webcam_detection(
            camera_id=args.camera_id,
            output_file=args.output
        )
    else:
        # Process video file
        if not os.path.exists(args.source):
            print(f"Error: Source file not found: {args.source}")
            sys.exit(1)
        
        app.process_video_file(
            video_path=args.source,
            output_path=args.output
        )

if __name__ == "__main__":
    main()