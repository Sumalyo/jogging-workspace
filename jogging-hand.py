"""
Hand-controlled robot jogging using video file (headless mode).

This demonstrates:
- Real-time hand tracking with OpenCV from video file
- Thread-safe integration of video feed with async robot control
- Continuous jogging based on hand position
- Safe velocity limits and dead zones

How it works:
- Plays a video file with hand detection (no preview window)
- Center zone = no movement (dead zone)
- Move hand left/right = jog robot TCP in X-axis
- Move hand up/down = jog robot TCP in Y-axis
- Press Ctrl+C to quit

Note: Uses color-based hand detection (works best with good lighting
and distinct hand color vs background).
"""

import asyncio
import logging
import threading
from queue import Queue
from typing import Optional
import os

import cv2
import numpy as np
import aiohttp

import nova
from nova import api, run_program
from nova.cell import virtual_controller
from nova.core.nova import Nova
from nova.program import ProgramPreconditions

# Load environment variables for token
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
controller_name = "urdatta10"
NOVA_API_URL = "https://bvyzqvgo.instance.wandelbots.io/api/v2"
NOVA_TOKEN = os.getenv("NOVA_API_KEY")  # Get token from .env file

# Video file path - UPDATE THIS to your video file
VIDEO_FILE = "hand_video.mp4"

# Hand tracking settings
DEAD_ZONE = 0.15  # Center area where no jogging occurs (0.0-0.5)
MAX_JOG_SPEED = 50.0  # mm/s maximum jogging speed
SMOOTHING_FACTOR = 0.3  # Lower = smoother but slower response (0.0-1.0)


class HandTracker:
    """Tracks hand position from video file using color detection (headless)."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Thread-safe position queue (normalized -1.0 to 1.0)
        self.position_queue: Queue[tuple[float, float, bool]] = Queue(maxsize=1)
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Smoothed position
        self.smooth_x = 0.0
        self.smooth_y = 0.0
        
        # Video properties
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 30
        
        # Skin color detection ranges (HSV)
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0

    def start(self) -> None:
        """Start the hand tracking thread."""
        if self.running:
            logger.warning("Hand tracker already running")
            return

        if not os.path.exists(self.video_path):
            raise RuntimeError(f"Video file not found: {self.video_path}")

        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Starting hand tracker with video: {self.video_path}")
        logger.info(f"Video: {self.frame_width}x{self.frame_height} @ {self.fps}fps ({total_frames} frames)")
        logger.info("Running in headless mode (no preview window)")
        logger.info("Press Ctrl+C to quit")
        
        self.running = True
        self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the hand tracking thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        
        # Print statistics
        if self.frame_count > 0:
            detection_rate = (self.detection_count / self.frame_count) * 100
            logger.info(f"Tracking stats: {self.detection_count}/{self.frame_count} frames with hand detected ({detection_rate:.1f}%)")
        
        logger.info("Hand tracker stopped")

    def _restart_video(self) -> None:
        """Restart video from beginning."""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            logger.info("Video restarted - looping playback")

    def _detect_hand(self, frame):
        """
        Detect hand in frame using color detection.
        
        Returns:
            (x, y, detected) tuple where x,y are normalized positions
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for skin color
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0, 0.0, False
        
        # Get largest contour (assume it's the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter out small detections (noise)
        area = cv2.contourArea(largest_contour)
        if area < 3000:  # Minimum area threshold
            return 0.0, 0.0, False
        
        # Get center of mass
        moments = cv2.moments(largest_contour)
        if moments["m00"] == 0:
            return 0.0, 0.0, False
        
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        
        # Normalize to -1.0 to 1.0
        norm_x = (center_x / self.frame_width - 0.5) * 2.0
        norm_y = (center_y / self.frame_height - 0.5) * 2.0
        
        return norm_x, norm_y, True

    def _tracking_loop(self) -> None:
        """Main tracking loop (runs in separate thread)."""
        try:
            while self.running:
                try:
                    ret, frame = self.cap.read()
                    
                    # Loop video if it ends
                    if not ret:
                        self._restart_video()
                        continue

                    self.frame_count += 1
                    hand_detected = False
                    control_x = 0.0
                    control_y = 0.0
                    
                    # Detect hand
                    raw_x, raw_y, hand_detected = self._detect_hand(frame)
                    
                    if hand_detected:
                        self.detection_count += 1
                        
                        # Log raw detection
                        logger.debug(f"Raw hand position: x={raw_x:.3f}, y={raw_y:.3f}")
                        
                        # Apply dead zone
                        if abs(raw_x) < DEAD_ZONE:
                            raw_x = 0.0
                        else:
                            raw_x = (abs(raw_x) - DEAD_ZONE) / (1.0 - DEAD_ZONE)
                            raw_x = raw_x if raw_x > 0 else -raw_x
                            raw_x = raw_x * (1.0 if raw_x > 0 else -1.0)
                        
                        if abs(raw_y) < DEAD_ZONE:
                            raw_y = 0.0
                        else:
                            raw_y = (abs(raw_y) - DEAD_ZONE) / (1.0 - DEAD_ZONE)
                            raw_y = raw_y if raw_y > 0 else -raw_y
                            raw_y = raw_y * (1.0 if raw_y > 0 else -1.0)
                        
                        # Smooth the values
                        self.smooth_x = (SMOOTHING_FACTOR * raw_x + 
                                        (1.0 - SMOOTHING_FACTOR) * self.smooth_x)
                        self.smooth_y = (SMOOTHING_FACTOR * raw_y + 
                                        (1.0 - SMOOTHING_FACTOR) * self.smooth_y)
                        
                        control_x = self.smooth_x
                        control_y = -self.smooth_y  # Invert Y
                        
                        logger.debug(f"Control output: x={control_x:.3f}, y={control_y:.3f}")
                    
                    # Update queue (non-blocking)
                    if not self.position_queue.full():
                        try:
                            self.position_queue.get_nowait()
                        except:
                            pass
                    self.position_queue.put((control_x, control_y, hand_detected))
                    
                    # Log status more frequently for debugging
                    if self.frame_count % self.fps == 0:  # Every second
                        speed_x = control_x * MAX_JOG_SPEED
                        speed_y = control_y * MAX_JOG_SPEED
                        status = "HAND DETECTED" if hand_detected else "NO HAND"
                        logger.info(f"Frame {self.frame_count}: {status} | Speed: X={speed_x:+.1f} Y={speed_y:+.1f} mm/s")
                    
                    # Small delay to match video FPS
                    import time
                    time.sleep(1.0 / self.fps)
                
                except cv2.error as e:
                    logger.error(f"OpenCV error: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in hand tracking: {e}")
            self.running = False


async def jog_control_loop(motion_group, tracker: HandTracker) -> None:
    """
    Main jogging control loop.
    
    Reads hand position from tracker and sends jog commands to robot.
    """
    logger.info("Starting jog control loop")
    
    last_x = 0.0
    last_y = 0.0
    mg_id = motion_group.motion_group_id
    
    jog_command_count = 0
    stop_command_count = 0
    
    # Create HTTP session with auth headers
    headers = {
        "Authorization": f"Bearer {NOVA_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    # API endpoints
    jog_url = f"{NOVA_API_URL}/cells/cell/motion-groups/{mg_id}/jogging"
    
    async with aiohttp.ClientSession(headers=headers) as session:
        while tracker.running:
            try:
                control_x, control_y, hand_detected = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: tracker.position_queue.get(timeout=0.05)
                )
                
                vel_x = control_x * MAX_JOG_SPEED
                vel_y = control_y * MAX_JOG_SPEED
                
                if abs(vel_x - last_x) > 1.0 or abs(vel_y - last_y) > 1.0 or not hand_detected:
                    if hand_detected and (abs(vel_x) > 0.1 or abs(vel_y) > 0.1):
                        # Jog in Cartesian mode with velocity vector
                        logger.info(f"Sending jog command: X={vel_x:.1f}, Y={vel_y:.1f} mm/s")
                        
                        payload = {
                            "mode": "cartesian",
                            "tcp_speed": [vel_x, vel_y, 0.0, 0.0, 0.0, 0.0]
                        }
                        
                        async with session.post(jog_url, json=payload) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                logger.error(f"Jog API error: {response.status} - {error_text}")
                            else:
                                jog_command_count += 1
                                logger.debug(f"Jog command successful")
                        
                        last_x = vel_x
                        last_y = vel_y
                    else:
                        # Stop jogging
                        logger.info("Sending stop jog command")
                        
                        async with session.delete(jog_url) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                logger.error(f"Stop jog API error: {response.status} - {error_text}")
                            else:
                                stop_command_count += 1
                                logger.debug(f"Stop jog command successful")
                        
                        last_x = 0.0
                        last_y = 0.0
                        
            except Exception as e:
                logger.error(f"Jog error: {e}", exc_info=True)
                await asyncio.sleep(0.01)
        
        # Stop jogging when exiting
        try:
            async with session.delete(jog_url) as response:
                logger.debug(f"Final stop jog: {response.status}")
        except Exception as e:
            logger.debug(f"Stop jogging error: {e}")
    
    logger.info(f"Jog control loop ended - Sent {jog_command_count} jog commands, {stop_command_count} stop commands")


@nova.program(
    id="hand_jog",
    name="Hand-Controlled Jogging",
    preconditions=ProgramPreconditions(
        controllers=[
            virtual_controller(
                name=controller_name,
                manufacturer=api.models.Manufacturer.UNIVERSALROBOTS,
                type=api.models.VirtualControllerTypes.UNIVERSALROBOTS_MINUS_UR5E,
            )
        ],
        cleanup_controllers=False,
    ),
)
async def hand_jog():
    """Control robot jogging with hand tracking from video file."""
    tracker = HandTracker(video_path=VIDEO_FILE)
    
    try:
        tracker.start()
        
        async with Nova() as nova_instance:
            cell = nova_instance.cell()
            controller = await cell.controller(controller_name)
            
            async with controller[0] as motion_group:
                logger.info("Robot ready for hand control!")
                logger.info(f"Motion Group ID: {motion_group.motion_group_id}")
                logger.info(f"Dead zone: {DEAD_ZONE * 100:.0f}%")
                logger.info(f"Max speed: {MAX_JOG_SPEED} mm/s")
                
                await jog_control_loop(motion_group, tracker)
                
    except KeyboardInterrupt:
        logger.info("\nShutting down hand jog...")
    except Exception as e:
        logger.error(f"Error in hand jog program: {e}", exc_info=True)
    finally:
        tracker.stop()


if __name__ == "__main__":
    run_program(hand_jog)