"""
PART 2: Configuration and Parameters
Copy this into your main file (tokyo_neural_nav.py)
"""

from PyQt6.QtGui import QColor
import json
import os  





"""
PART 3: Enhanced DQN Network Architecture
Copy this into your main file (tokyo_neural_nav.py)
"""

import torch.nn as nn

# ==========================================
# ENHANCED DQN - ADDED 1 FC LAYER (Assignment 2)
# ==========================================

class EnhancedDrivingDQN(nn.Module):
    """
    Enhanced Deep Q-Network for Assignment 2
    
    ORIGINAL ARCHITECTURE (5 layers):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Input(9) → FC(128) → ReLU
             → FC(256) → ReLU
             → FC(256) → ReLU
             → FC(128) → ReLU
             → Output(5)
    
    ENHANCED ARCHITECTURE (6 layers):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Input(9) → FC(128) → ReLU
             → FC(256) → ReLU
             → FC(256) → ReLU
             → FC(256) → ReLU  ← ADDED THIS LAYER
             → FC(128) → ReLU
             → Output(5)
    
    WHY ADD THIS LAYER?
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. Increased network capacity for complex navigation
    2. Better feature extraction from 7 sensor inputs
    3. Improved generalization across different map areas
    4. More non-linear transformations = better learning
    
    INPUT:
    - 7 sensor values (brightness 0-1)
    - 1 angle to target (normalized -1 to 1)
    - 1 distance to target (normalized 0-1)
    Total: 9 features
    
    OUTPUT:
    - Q-values for 5 actions:
      0: Turn left (TURN_SPEED)
      1: Go straight
      2: Turn right (TURN_SPEED)
      3: Sharp left (SHARP_TURN)
      4: Sharp right (SHARP_TURN)
    """
    
    def __init__(self, input_dim, output_dim):
        super(EnhancedDrivingDQN, self).__init__()
        
        self.net = nn.Sequential(
            # Layer 1: Input → 128
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            
            # Layer 2: 128 → 256 (expand)
            nn.Linear(128, 256),
            nn.ReLU(),
            
            # Layer 3: 256 → 256 (maintain)
            nn.Linear(256, 256),
            nn.ReLU(),
            
            # ═══════════════════════════════════════════
            # Layer 4: 256 → 256 (ADDED - Assignment 2)
            # ═══════════════════════════════════════════
            nn.Linear(256, 256),
            nn.ReLU(),
            
            # Layer 5: 256 → 128 (compress)
            nn.Linear(256, 128),
            nn.ReLU(),
            
            # Layer 6: 128 → Output
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 9)
        
        Returns:
            Q-values tensor of shape (batch_size, 5)
        """
        return self.net(x)
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==========================================
# NETWORK COMPARISON
# ==========================================

def compare_networks():
    """
    Compare original vs enhanced network
    Run this to see the difference
    """
    # Original network (5 layers)
    original = nn.Sequential(
        nn.Linear(9, 128), nn.ReLU(),
        nn.Linear(128, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 5)
    )
    
    # Enhanced network (6 layers)
    enhanced = EnhancedDrivingDQN(9, 5)
    
    orig_params = sum(p.numel() for p in original.parameters())
    enh_params = enhanced.count_parameters()
    
    print("NETWORK COMPARISON:")
    print("━" * 50)
    print(f"Original Network:  {orig_params:,} parameters")
    print(f"Enhanced Network:  {enh_params:,} parameters")
    print(f"Difference:        +{enh_params - orig_params:,} parameters")
    print(f"Increase:          {(enh_params/orig_params - 1)*100:.1f}%")
    print("━" * 50)


if __name__ == "__main__":
    # Test the network
    compare_networks()
    
    # Create instance
    model = EnhancedDrivingDQN(input_dim=9, output_dim=5)
    print(f"\n✅ Enhanced DQN created")
    print(f"   Total parameters: {model.count_parameters():,}")
    print(f"   Layers: 6 (added 1 FC layer)")


"""
PART 4: Car Brain - Reinforcement Learning Logic
Copy this into your main file (tokyo_neural_nav.py)
"""

import math
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor

# Import from Part 2 and Part 3
# from config import *
# from network import EnhancedDrivingDQN


class CarBrain:
    """
    The brain of the autonomous car - handles RL logic
    """
    
    def __init__(self, map_image, config):
        """
        Initialize the car's brain
        
        Args:
            map_image: QImage of the map
            config: Configuration object with parameters
        """
        self.map = map_image
        self.w = map_image.width()
        self.h = map_image.height()
        self.config = config
        
        # RL Components
        self.input_dim = 9  # 7 sensors + angle + distance
        self.n_actions = 5  # left, straight, right, sharp_left, sharp_right
        
        # Networks
        self.policy_net = EnhancedDrivingDQN(self.input_dim, self.n_actions)
        self.target_net = EnhancedDrivingDQN(self.input_dim, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LR)
        
        # Experience Replay Buffers
        self.memory = deque(maxlen=10000)
        self.priority_memory = deque(maxlen=3000)
        self.current_episode_buffer = []
        self.episode_scores = deque(maxlen=100)
        
        # Training stats
        self.steps = 0
        self.epsilon = 1.0  # FIXED: Start with full exploration
        self.consecutive_crashes = 0
        self.episode_count = 0
        self.successful_episodes = 0
        self.total_steps = 0
        
        # Car state
        # self.start_pos = QPointF(config.START_POS[0], config.START_POS[1])
        # self.car_pos  = QPointF(config.START_POS[0], config.START_POS[1])
        
        # Placeholder; actual start will be injected by controller
        self.start_pos = QPointF(0, 0)
        self.car_pos = QPointF(0, 0)


        self.car_angle = 0
        self.alive = True
        self.score = 0
        
        # Sensors
        self.sensor_coords = []
        self.prev_dist = None
        
        # Targets
        self.targets = []
        self.current_target_idx = 0
        self.target_pos = QPointF(0,0)
        self.targets_reached = 0
        # ✅ ADD: Curriculum learning
        self.active_targets = 1  # Start with only Target 1
        self.target_unlock_threshold = 20  # Successful episodes needed
        # ✅ ADD: Episode length management
        # self.max_steps_per_target = 500  # Steps allowed per target
        self.steps_current_target = 0

        # ✅ ADD: Separate buffers per target
        self.target_memories = [deque(maxlen=3000) for _ in range(3)]
        
        # Visualization
        self.path_trail = deque(maxlen=200)
        self.target_reach_counts = [0, 0, 0]  # Initialize BEFORE loading
        self.load_model() 

    def save_model(self, filename="last_brain.pth"):
        """Save the current neural network weights AND training state"""
        checkpoint = {
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'successful_episodes': self.successful_episodes,
            'total_steps': self.total_steps,
            # ✅ ADD: Save curriculum learning state
            'active_targets': self.active_targets,
            'target_reach_counts': self.target_reach_counts
        }
        torch.save(checkpoint, filename)
        print(f"💾 Saved checkpoint to {filename}")
        print(f"   Epsilon: {self.epsilon:.3f} | Episodes: {self.episode_count}")
        print(f"   Active targets: {self.active_targets}/3")

    def load_model(self, filename="last_brain.pth"):
        """Load neural network weights AND training state"""
        if os.path.exists(filename):
            try:
                checkpoint = torch.load(filename)
                
                if isinstance(checkpoint, dict):
                    # New format with metadata
                    self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    # Restore training state
                    self.epsilon = checkpoint.get('epsilon', 0.15)
                    self.episode_count = checkpoint.get('episode_count', 0)
                    self.successful_episodes = checkpoint.get('successful_episodes', 0)
                    self.total_steps = checkpoint.get('total_steps', 0)
                    
                    # ✅ RESTORE: Curriculum learning state
                    self.active_targets = checkpoint.get('active_targets', 1)
                    self.target_reach_counts = checkpoint.get('target_reach_counts', [0, 0, 0])
                    
                    print(f"✅ Loaded checkpoint from {filename}")
                    print(f"   Epsilon: {self.epsilon:.3f}")
                    print(f"   Episodes: {self.episode_count}")
                    print(f"   Successful: {self.successful_episodes}")
                    print(f"   Active targets: {self.active_targets}/3")
                    print(f"   Target reaches: T1={self.target_reach_counts[0]}, T2={self.target_reach_counts[1]}, T3={self.target_reach_counts[2]}")
                else:
                    # Old format (just state dict)
                    self.policy_net.load_state_dict(checkpoint)
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    self.epsilon = 0.15
                    print(f"✅ Loaded model (old format)")
                    
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                self.epsilon = 1.0
        else:
            print("🆕 No saved model found, starting fresh")
            self.epsilon = 1.0

    
    def set_start_pos(self, point):
        """Set the starting position"""
        self.start_pos = QPointF(point.x(), point.y())
        self.car_pos = QPointF(point.x(), point.y())
    
    def add_target(self, point):
        """Add a target waypoint"""
        self.targets.append(QPointF(point.x(), point.y()))
        if len(self.targets) == 1:
            self.target_pos = self.targets[0]
            self.current_target_idx = 0
    
    def reset(self):
        """Reset for a new episode"""
        self.alive = True
        self.score = 0
        self.car_pos = QPointF(self.start_pos.x(), self.start_pos.y())
        self.car_angle = random.randint(0, 360)

        self.current_target_idx = 0
        self.targets_reached = 0
        self.path_trail.clear()
        self.steps_current_target = 0  # ✅ ADD: Reset step counter



        # ✅ Unlock more targets based on success
        if self.successful_episodes >= self.target_unlock_threshold:
            self.active_targets = 2
        if self.successful_episodes >= self.target_unlock_threshold * 2:
            self.active_targets = 3
        
        if len(self.targets) > 0:
            self.target_pos = self.targets[0]
        
        state, dist = self.get_state()
        self.prev_dist = dist
        self.episode_count += 1

        # ✅ ONLY LOG EVERY 100 EPISODES
        if self.episode_count % 100 == 0:
            print(f"\n🔄 EPISODE {self.episode_count}")
            print(f"   Active targets: {self.active_targets}")
            print(f"   Success rate: {self.successful_episodes}/{self.episode_count}")
            print(f"   Epsilon: {self.epsilon:.3f}")
    
        
        return state
    
    def switch_to_next_target(self):
        """Move to next target in sequence"""
        old_active = self.active_targets

        # ✅ ADD THIS - Update curriculum immediately
        if self.successful_episodes >= self.target_unlock_threshold * 2:
            self.active_targets = 3
        elif self.successful_episodes >= self.target_unlock_threshold:
            self.active_targets = 2

        # ✅ FIX: Boost epsilon when unlocking new curriculum stage
        # ✅ FIX: Boost epsilon when curriculum advances
        if self.active_targets > old_active:
            if self.active_targets == 2:
                self.epsilon = max(self.epsilon, 0.35)
                print(f"🔓 TARGET 2 UNLOCKED! Epsilon: {self.epsilon:.3f}")
            elif self.active_targets == 3:
                self.epsilon = max(self.epsilon, 0.40)
                print(f"🔓 TARGET 3 UNLOCKED! Epsilon: {self.epsilon:.3f}")

        # ✅ Check if we can switch to next target
        if self.current_target_idx < min(len(self.targets) - 1, self.active_targets - 1):
            old_target_idx = self.current_target_idx
            self.current_target_idx += 1
            
            # ✅ CRITICAL: Update target_pos to the new target
            self.target_pos = self.targets[self.current_target_idx]
            
            self.targets_reached += 1
            self.steps_current_target = 0
            self.path_trail.clear()
            
            #✅ KEEP: This is important progress info
            print(f"   ➡️ Moving to Target {self.current_target_idx + 1}")
            

        # ✅ Only switch to targets that are unlocked
        # if self.current_target_idx < min(len(self.targets) - 1, self.active_targets - 1):
        #     self.current_target_idx += 1
        #     self.target_pos = self.targets[self.current_target_idx]
        #     self.targets_reached += 1
        #     self.steps_current_target = 0  # ✅ RESET steps for new target
        #     self.path_trail.clear() 
            # ✅ CRITICAL: reset position between curriculum stages
            # self.car_pos = QPointF(self.start_pos.x(), self.start_pos.y())
            # self.car_angle = random.randint(0, 360)

            # ✅ Boost exploration for harder targets
            # if self.current_target_idx >= 1 and self.epsilon < 0.30:
            #     self.epsilon = 0.30


            return True
        return False  # End episode when reaching curriculum limit
    
    def get_state(self):
        """
        Get current state from sensors
        
        Returns:
            state: numpy array of 10 features
            distance: distance to current target
        """

        # State layout:
        # [0-6] sensors
        # [7]   normalized angle to target
        # [8]   normalized distance to target
        # [9]   normalized distance to previous target

        sensor_vals = []
        self.sensor_coords = []
        
        # 7 sensors at different angles
        angles = [-45, -30, -15, 0, 15, 30, 45]
        
        for angle_offset in angles:
            rad = math.radians(self.car_angle + angle_offset)
            sx = self.car_pos.x() + math.cos(rad) * self.config.SENSOR_DIST
            sy = self.car_pos.y() + math.sin(rad) * self.config.SENSOR_DIST
            self.sensor_coords.append(QPointF(sx, sy))
            
            val = 0.0
            if 0 <= sx < self.w and 0 <= sy < self.h:
                c = QColor(self.map.pixel(int(sx), int(sy)))
                brightness = (c.red() + c.green() + c.blue()) / 3.0
                val = brightness / 255.0
            sensor_vals.append(val)
        
        # Calculate angle and distance to target
        dx = self.target_pos.x() - self.car_pos.x()
        dy = self.target_pos.y() - self.car_pos.y()
        dist = math.sqrt(dx*dx + dy*dy)
        
        rad_to_target = math.atan2(dy, dx)
        angle_to_target = math.degrees(rad_to_target)
        
        angle_diff = (angle_to_target - self.car_angle) % 360
        if angle_diff > 180:
            angle_diff -= 360
        
        # Normalize
        norm_dist = min(dist / 800.0, 1.0)
        norm_angle = angle_diff / 180.0

        # # ✅ ADD: Distance to previous target (helps with navigation memory)
        # if self.current_target_idx > 0:
        #     prev_target = self.targets[self.current_target_idx - 1]
        #     dx_prev = prev_target.x() - self.car_pos.x()
        #     dy_prev = prev_target.y() - self.car_pos.y()
        #     dist_to_prev = math.sqrt(dx_prev*dx_prev + dy_prev*dy_prev)
        #     norm_dist_prev = min(dist_to_prev / 800.0, 1.0)
        # else:
        #     norm_dist_prev = 0
         # ✅ REDUCE: Only log every 500 steps (was 100)
        if self.total_steps % 500 == 0:
            print(f"[State] Target {self.current_target_idx+1} | "
                f"Dist: {dist:.1f} | Angle: {angle_diff:.1f}° | "
                f"Epsilon: {self.epsilon:.3f}")
            
        state = sensor_vals + [norm_angle, norm_dist] #,norm_dist_prev
        return np.array(state, dtype=np.float32), dist
    
    def step(self, action):
        """
        Take action and return next state, reward, done
        
        Args:
            action: 0-4 (left, straight, right, sharp_left, sharp_right)
        
        Returns:
            next_state, reward, done
        """
        # Record path
        self.path_trail.append(QPointF(self.car_pos.x(), self.car_pos.y()))
        self.steps_current_target += 1

        force_terminate = False


        # ❌ Kill episodes that make no progress
        max_steps = 600 + self.current_target_idx * 200
        if self.steps_current_target > max_steps:
            force_terminate = True


            
         

        
        # Apply action
        turn = 0
        if action == 0:  # Left
            turn = -self.config.TURN_SPEED
        elif action == 1:  # Straight
            turn = 0
        elif action == 2:  # Right
            turn = self.config.TURN_SPEED
        elif action == 3:  # Sharp left
            turn = -self.config.SHARP_TURN
        elif action == 4:  # Sharp right
            turn = self.config.SHARP_TURN
        
        self.car_angle += turn
        rad = math.radians(self.car_angle)
        
        # Move forward
        new_x = self.car_pos.x() + math.cos(rad) * self.config.SPEED
        new_y = self.car_pos.y() + math.sin(rad) * self.config.SPEED
        self.car_pos = QPointF(new_x, new_y)
        
        # Get next state
        next_state, dist = self.get_state()
        
        reward = -0.03 # -0.05  # Small penalty per step
        done = False
        
        # Check collision
        car_center_val = self.check_pixel(self.car_pos.x(), self.car_pos.y())

        # For grid map: bright (>0.8) = street (safe), dark (<0.8) = building/obstacle
        if car_center_val < 0.3:  # Hit building/obstacle or went off street #reduced to 0.3 now more forgiving
            reward = -100 #increased from 50, higher penalty
            done = True
            self.alive = False
        
        elif dist < 30:  # Reached target
            # ✅ ADD: Increment target reach counter
            self.target_reach_counts[self.current_target_idx] += 1

            base_reward = 100
            target_bonus = (self.current_target_idx + 1) * 50
            reward = base_reward + target_bonus
            
            print(f"🎯 REACHED Target {self.current_target_idx + 1}! (Episode {self.episode_count})")

            
            has_next = self.switch_to_next_target()
            if has_next:
                reward += 20
                done = False  # ✅ CRITICAL: Continue episode
                
                # ✅ FIX: Update prev_dist to new target distance
                next_state, new_dist = self.get_state()
                self.prev_dist = new_dist
                
               
            else:
                reward += 200
                done = True
                self.successful_episodes += 1
                print(f"   🏁 ALL TARGETS COMPLETE!")
       
        # elif dist < 30:  # Reached target
        #     # ✅ ADD: Increment target reach counter
        #     self.target_reach_counts[self.current_target_idx] += 1
        
        #     base_reward = 100
        #     target_bonus = (self.current_target_idx + 1) * 50  # +50, +100, +150 # higher reward for later targets
        #     reward = base_reward + target_bonus
        #     has_next = self.switch_to_next_target()
        #     if has_next:
        #         reward += 20
        #         done = False
        #         _, new_dist = self.get_state()
        #         self.prev_dist = new_dist
        #     else:
        #         reward+=200 #higher reward for completing all 3 targets
        #         done = True
        #         self.successful_episodes += 1

        # 🎯 NEAR-MISS BONUS
        elif dist < 60:   # 2 * HIT_RADIUS (30 * 2)
            reward += 2.0
            # do NOT set done = True
        else:
            # ✅ IMPROVED: Stronger reward for getting closer
            if self.prev_dist is not None:
                diff = self.prev_dist - dist
                progress_scale = 10 * (1.5 ** self.current_target_idx)
                # progress_scale = 10 + (self.current_target_idx * 5)
                reward += diff * progress_scale

            # ✅ EXTRA: Move away from start for later targets
            # if self.current_target_idx >= 1:
            #     dx_start = self.car_pos.x() - self.start_pos.x()
            #     dy_start = self.car_pos.y() - self.start_pos.y()
            #     dist_from_start = math.sqrt(dx_start*dx_start + dy_start*dy_start)
            #     reward += (dist_from_start / 800.0) * 2.0
            
            # ✅ IMPROVED: Bigger bonus for staying on bright roads
            reward += next_state[3] * 5  # Increased from 2 to 5
            
            # ✅ IMPROVED: Bigger penalty for moving away
            if self.prev_dist is not None and dist > self.prev_dist:
                reward -= 1.0  # Increased from -0.1

            # 🎯 Encourage commitment toward Target 3 (directional shaping)
            if self.current_target_idx == 2:
                angle_alignment = 1.0 - abs(next_state[7])  # normalized angle to target
                # reward += angle_alignment * 2.0
                reward += max(0.0, angle_alignment) * 2.0


            # 🔴 LOOP / STAGNATION PENALTY
            # -------------------------
            if len(self.path_trail) >= 20:
                dx = self.path_trail[-1].x() - self.path_trail[-20].x()
                dy = self.path_trail[-1].y() - self.path_trail[-20].y()
                if math.sqrt(dx*dx + dy*dy) < 15:
                    reward -= 20.0
                    done = True          # kill looping episodes
                    self.alive = False

        if force_terminate:
            reward = -50
            done = True
            self.alive = False

        self.prev_dist = dist

        self.score += reward
        self.total_steps += 1

        # ✅ ADD: Periodic logging for debugging
        # if self.total_steps % 200 == 0:
        #     print(f"[Debug] Target {self.current_target_idx+1} | "
        #         f"Dist: {dist:.1f} | "
        #         f"Reward: {reward:.2f} | "
        #         f"Epsilon: {self.epsilon:.3f} | "
        #         f"Steps: {self.steps_current_target}")
            
        
        return next_state, reward, done
    
    def check_pixel(self, x, y):
        """Check if pixel is navigable (bright = road)"""
        if 0 <= x < self.w and 0 <= y < self.h:
            c = QColor(self.map.pixel(int(x), int(y)))
            return ((c.red() + c.green() + c.blue()) / 3.0) / 255.0
        return 0.0
    
    def store_experience(self, experience):
        """Store experience for replay"""
        self.current_episode_buffer.append(experience)
        # ✅ ADD: Also store in target-specific buffer
        target_idx = self.current_target_idx
        if target_idx < len(self.target_memories):
            self.target_memories[target_idx].append(experience)
    
    def finalize_episode(self, episode_reward):
        """Store episode in appropriate buffer"""
        if len(self.current_episode_buffer) == 0:
            return
        
        self.episode_scores.append(episode_reward)
        
        if not self.alive:
            self.consecutive_crashes += 1
        else:
            self.consecutive_crashes = 0
        
        # Prioritized replay: good episodes go to priority buffer
        if episode_reward > 0:
            for exp in self.current_episode_buffer:
                self.priority_memory.append(exp)
        else:
            for exp in self.current_episode_buffer:
                self.memory.append(exp)
        
        self.current_episode_buffer = []
        if self.episode_count % 50 == 0:
            self.save_model()
    
    def optimize(self):
        """
        Train the network on a batch of experiences
        Uses prioritized replay
        """
        total_memory = len(self.memory) + len(self.priority_memory)
        if total_memory < self.config.BATCH_SIZE:
            return 0
        
        # ✅ NEW: Sample equally from each target buffer
        batch = []
        samples_per_target = max(8, self.config.BATCH_SIZE // 3)

        
        for target_mem in self.target_memories:
            if len(target_mem) >= samples_per_target:
                batch.extend(random.sample(target_mem, samples_per_target))
            else:
                batch.extend(list(target_mem))
        
        # Fill remaining with priority memory
        remaining = self.config.BATCH_SIZE - len(batch)
        if remaining > 0 and len(self.priority_memory) >= remaining:
            batch.extend(random.sample(self.priority_memory, remaining))
    
        # 🎯 Bias sampling toward CURRENT target
        current_mem = self.target_memories[self.current_target_idx]
        if len(current_mem) >= 8:
            batch.extend(random.sample(current_mem, 8))
        
        if len(batch) < self.config.BATCH_SIZE // 2:
            return 0
        
        # Prepare tensors
        s, a, r, ns, d = zip(*batch)
        s = torch.FloatTensor(np.array(s))
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        ns = torch.FloatTensor(np.array(ns))
        d = torch.FloatTensor(d).unsqueeze(1)
        
        # Compute loss
        q = self.policy_net(s).gather(1, a)
        next_q = self.target_net(ns).max(1)[0].detach().unsqueeze(1)
        target = r + self.config.GAMMA * next_q * (1 - d)
        
        loss = nn.MSELoss()(q, target)
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        # ✅ NEW: Slower decay + target-dependent minimum
        # ✅ FIX: Minimum epsilon depends on curriculum stage
        if self.active_targets == 1:
            min_epsilon = 0.15  # Target 1: can exploit more
        elif self.active_targets == 2:
            min_epsilon = 0.25  # Target 1→2: needs more exploration
        else:  # active_targets == 3
            min_epsilon = 0.30  # Target 1→2→3: needs even more exploration
        
        if self.epsilon > min_epsilon:
            self.epsilon *= 0.9995  # Slower decay (was 0.997)
            
        return loss.item()
    
    def update_target_network(self):
        """Soft update of target network"""
        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.config.TAU * policy_param.data + 
                (1.0 - self.config.TAU) * target_param.data
            )

"""
PART 5: Visual Components (UI Widgets)
Copy this into your main file (tokyo_neural_nav.py)
"""

from PyQt6.QtWidgets import QWidget, QGraphicsItem
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath, QFont
from PyQt6.QtCore import Qt, QPointF, QRectF

# Import from Part 2
# from config import C_PANEL, C_ACCENT, C_SUCCESS, etc.


# ==========================================
# REWARD CHART WIDGET
# ==========================================

class RewardChart(QWidget):
    """
    Displays reward history over episodes
    Shows both raw rewards and moving average
    """
    
    def __init__(self, bg_color, accent_color, success_color):
        super().__init__()
        self.setMinimumHeight(150)
        self.setStyleSheet(f"background-color: {bg_color.name()}; border-radius: 5px;")
        
        self.bg_color = bg_color
        self.accent_color = accent_color
        self.success_color = success_color
        
        self.scores = []
        self.max_points = 50

    def update_chart(self, new_score):
        """Add new score and update display"""
        self.scores.append(new_score)
        if len(self.scores) > self.max_points:
            self.scores.pop(0)
        self.update()

    def paintEvent(self, event):
        """Draw the chart"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        painter.fillRect(0, 0, w, h, self.bg_color)
        
        if len(self.scores) < 2:
            return

        min_val = min(self.scores)
        max_val = max(self.scores)
        if max_val == min_val: 
            max_val += 1
        
        # Plot raw scores
        points = []
        step_x = w / (self.max_points - 1)
        
        for i, score in enumerate(self.scores):
            x = i * step_x
            ratio = (score - min_val) / (max_val - min_val)
            y = h - (ratio * (h * 0.8) + (h * 0.1))
            points.append(QPointF(x, y))

        path = QPainterPath()
        path.moveTo(points[0])
        for p in points[1:]:
            path.lineTo(p)
            
        pen = QPen(self.accent_color, 2)
        painter.setPen(pen)
        painter.drawPath(path)
        
        # Plot moving average
        if len(self.scores) >= 2:
            avg_points = []
            window_size = 10
            
            for i in range(len(self.scores)):
                start_idx = max(0, i - window_size + 1)
                avg_score = sum(self.scores[start_idx:i+1]) / (i - start_idx + 1)
                
                x = i * step_x
                ratio = (avg_score - min_val) / (max_val - min_val)
                y = h - (ratio * (h * 0.8) + (h * 0.1))
                avg_points.append(QPointF(x, y))
            
            if len(avg_points) > 1:
                avg_path = QPainterPath()
                avg_path.moveTo(avg_points[0])
                for p in avg_points[1:]:
                    avg_path.lineTo(p)
                
                avg_pen = QPen(self.success_color, 3)
                painter.setPen(avg_pen)
                painter.drawPath(avg_path)
        
        # Zero line
        if min_val < 0 and max_val > 0:
            zero_ratio = (0 - min_val) / (max_val - min_val)
            y_zero = h - (zero_ratio * (h * 0.8) + (h * 0.1))
            painter.setPen(QPen(QColor(255, 255, 255, 50), 1, Qt.PenStyle.DashLine))
            painter.drawLine(0, int(y_zero), w, int(y_zero))


# ==========================================
# SENSOR ITEM (GRAPHICS)
# ==========================================

class SensorItem(QGraphicsItem):
    """
    Visual representation of car sensors
    Shows pulsing dot - green if detecting road, red if obstacle
    """
    
    def __init__(self, sensor_on_color, sensor_off_color):
        super().__init__()
        self.setZValue(90)
        
        self.sensor_on = sensor_on_color
        self.sensor_off = sensor_off_color
        
        self.pulse = 0
        self.pulse_speed = 0.3
        self.is_detecting = True
        
    def set_detecting(self, detecting):
        """Update detection state"""
        self.is_detecting = detecting
        self.update()
    
    def boundingRect(self):
        return QRectF(-4, -4, 8, 8)
    
    def paint(self, painter, option, widget):
        """Draw pulsing sensor dot"""
        self.pulse += self.pulse_speed
        if self.pulse > 1.0:
            self.pulse = 0
        
        if self.is_detecting:
            color = self.sensor_on
            outer_alpha = int(150 * (1 - self.pulse))
        else:
            color = self.sensor_off
            outer_alpha = int(200 * (1 - self.pulse))
        
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Outer pulse
        outer_size = 3 + (2 * self.pulse)
        outer_color = QColor(color)
        outer_color.setAlpha(outer_alpha)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(outer_color))
        painter.drawEllipse(QPointF(0, 0), outer_size, outer_size)
        
        # Inner dot
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(0, 0), 2, 2)


# ==========================================
# CAR ITEM (GRAPHICS)
# ==========================================

class CarItem(QGraphicsItem):
    """
    Visual representation of the autonomous car
    """
    
    def __init__(self, car_width, car_height, accent_color):
        super().__init__()
        self.setZValue(100)
        
        self.car_w = car_width
        self.car_h = car_height
        
        self.brush = QBrush(accent_color)
        self.pen = QPen(Qt.GlobalColor.white, 2)

    def boundingRect(self):
        return QRectF(-self.car_w/2, -self.car_h/2, self.car_w, self.car_h)

    def paint(self, painter, option, widget):
        """Draw the car"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Car body
        painter.setBrush(self.brush)
        painter.setPen(self.pen)
        painter.drawRoundedRect(self.boundingRect(), 2, 2)
        
        # Front indicator
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawRect(int(self.car_w/2)-2, -3, 2, 6)


# ==========================================
# TARGET ITEM (GRAPHICS)
# ==========================================

class TargetItem(QGraphicsItem):
    """
    Visual representation of target waypoints
    Shows pulsing circle with label
    """
    
    def __init__(self, color, label, is_active=True):
        super().__init__()
        self.setZValue(50)
        
        self.color = color
        self.label = label
        self.is_active = is_active
        
        self.pulse = 0
        self.growing = True

    def set_active(self, active):
        """Update active state"""
        self.is_active = active
        self.update()

    def boundingRect(self):
        return QRectF(-40, -40, 80, 90)

    def paint(self, painter, option, widget):
        """Draw the target"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.is_active:
            # Pulsing animation
            if self.growing:
                self.pulse += 0.5
                if self.pulse > 10: 
                    self.growing = False
            else:
                self.pulse -= 0.5
                if self.pulse < 0: 
                    self.growing = True
            
            # Outer pulse - BIGGER
            r = 25 + self.pulse
            painter.setPen(Qt.PenStyle.NoPen)
            outer_color = QColor(self.color)
            outer_color.setAlpha(120)
            painter.setBrush(QBrush(outer_color)) 
            painter.drawEllipse(QPointF(0, 0), r, r)

            # Inner circle - BIGGER
            painter.setBrush(QBrush(self.color)) 
            painter.setPen(QPen(Qt.GlobalColor.white, 3))
            painter.drawEllipse(QPointF(0, 0), 20, 20)
        else:
            # Dimmed (already reached) - BIGGER
            dimmed_color = QColor(self.color)
            dimmed_color.setAlpha(120)
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.setBrush(QBrush(dimmed_color))
            painter.drawEllipse(QPointF(0, 0), 15, 15)
        
        # Label text
        # Label text with BLACK BACKGROUND
        painter.setFont(QFont("Arial", 11, QFont.Weight.Bold))

        # Draw black background box
        text_rect = QRectF(-35, 25, 70, 24)
        painter.fillRect(text_rect, QColor(0, 0, 0, 200))

        # Draw white text on black background
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.drawText(
            QRectF(-35, 27, 70, 24), 
            Qt.AlignmentFlag.AlignCenter, 
            self.label
        )

# ==========================================
# PATH TRAIL ITEM (GRAPHICS)
# ==========================================

class PathTrailItem(QGraphicsItem):
    """
    Shows the path the car has taken
    Fades from recent (bright) to old (dim)
    """
    
    def __init__(self, trail_color):
        super().__init__()
        self.setZValue(40)
        self.trail_color = trail_color
        self.points = []
    
    def set_points(self, points):
        """Update trail points"""
        self.points = list(points)
        self.update()
    
    def boundingRect(self):
        if not self.points:
            return QRectF(0, 0, 0, 0)
        
        xs = [p.x() for p in self.points]
        ys = [p.y() for p in self.points]
        
        return QRectF(
            min(xs) - 5, 
            min(ys) - 5,
            max(xs) - min(xs) + 10,
            max(ys) - min(ys) + 10
        )
    
    def paint(self, painter, option, widget):
        """Draw fading trail"""
        if len(self.points) < 2:
            return
        
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw trail with fading alpha
        for i in range(len(self.points) - 1):
            # Alpha increases toward recent points
            alpha = int(255 * (i / len(self.points)))
            color = QColor(self.trail_color)
            color.setAlpha(alpha)
            
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawLine(self.points[i], self.points[i + 1])


# ==========================================
# HELPER FUNCTION
# ==========================================

def create_visual_components(config):
    """
    Factory function to create all visual components
    
    Args:
        config: Configuration object with colors
    
    Returns:
        Dictionary of component classes
    """
    return {
        'chart': lambda: RewardChart(
            config.C_PANEL, 
            config.C_ACCENT, 
            config.C_SUCCESS
        ),
        'sensor': lambda: SensorItem(
            config.C_SENSOR_ON, 
            config.C_SENSOR_OFF
        ),
        'car': lambda: CarItem(
            config.CAR_WIDTH, 
            config.CAR_HEIGHT, 
            config.C_ACCENT
        ),
        'target': lambda color, label, active=True: TargetItem(
            color, label, active
        ),
        'trail': lambda: PathTrailItem(
            QColor(233, 69, 96, 80)
        )
    }

"""
PART 6: Main Application


BEFORE RUNNING:
1. Run: python map_generator.py (creates the map)
2. Run: python neural_nav.py (starts training)
"""

import sys
import os
import random
import torch
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *


# ==========================================
# CONFIGURATION CLASS
# ==========================================

class Config:
    """Configuration holder"""
    def __init__(self):
        # Colors
        self.C_BG_DARK = QColor("#1a1a2e")
        self.C_PANEL = QColor("#16213e")
        self.C_INFO_BG = QColor("#0f3460")
        self.C_ACCENT = QColor("#e94560")
        self.C_TEXT = QColor("#eaeaea")
        self.C_SUCCESS = QColor("#06ffa5")
        self.C_FAILURE = QColor("#ff006e")
        self.C_SENSOR_ON = QColor("#06ffa5")
        self.C_SENSOR_OFF = QColor("#ff006e")
        
        # Physics (FIXED)
        self.CAR_WIDTH = 45        # Smaller car for grid streets
        self.CAR_HEIGHT = 22
        self.SENSOR_DIST = 100 #60      # Shorter sensors for grid
        self.SPEED = 2    #2         # Slower speed for precise grid navigation
        self.TURN_SPEED = 8 #8        # Sharper turns for 90° grid
        self.SHARP_TURN = 20   #20     # Even sharper for tight corners

        self.SENSOR_ANGLE = 15
      
        
        # RL (FIXED)
        self.BATCH_SIZE = 64
        self.GAMMA = 0.99
        self.LR = 0.001
        self.TAU = 0.005
        self.MAX_CONSECUTIVE_CRASHES = 5
        
        # Targets
        self.TARGET_LABELS = ["Target 1", "Target 2", "Target 3"]

        self.TARGET_COLORS = [
            QColor(0, 255, 255),
            QColor(255, 100, 255),
            QColor(255, 215, 0),
        ]

        # Start/targets will be read from map metadata
        self.START_POS = None
        self.PRESET_TARGETS = []

        # Match the generator
        # self.BLOCK_SIZE = 50
        # self.STREET_WIDTH = 30
        # cell_size = self.BLOCK_SIZE + self.STREET_WIDTH

        # def street_to_pixel(sx, sy):
        #     px = sx * cell_size + self.STREET_WIDTH // 2
        #     py = sy * cell_size + self.STREET_WIDTH // 2
        #     return (px, py)

        # self.START_POS = street_to_pixel(2, 2)
        
        # # Street coordinates from map_generator.py
        # street_targets = [(8, 2), (2, 8), (8, 8)]
        # self.PRESET_TARGETS = [street_to_pixel(sx, sy) for sx, sy in street_targets]


# ==========================================
# MAIN APPLICATION WINDOW
# ==========================================

class NeuralNav(QMainWindow):
    """
    Main application window for Tokyo Metro Navigation
    """
    
    def __init__(self):
        super().__init__()
        
        self.config = Config()
        self.setup_window()
        self.setup_ui()
        self.load_map()
        self.setup_preset_targets()
        
        # Timers
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.game_loop)
        
        # ✅ NEW CODE (REPLACE WITH THIS):
        self.log("✅ Grid Navigation initialized")
        self.log("📍 3 targets preset: Target 1 → Target 2 → Target 3")
        self.log("🎯 Press START to begin training")
    
    def setup_window(self):
        """Configure main window"""
        self.setWindowTitle("Grid Navigation - Enhanced Neural Navigation")
        self.resize(1400, 900)
        
        # Apply stylesheet
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {self.config.C_BG_DARK.name()}; }}
            QLabel {{ 
                color: {self.config.C_TEXT.name()}; 
                font-family: 'Segoe UI'; 
                font-size: 13px; 
            }}
            QPushButton {{ 
                background-color: {self.config.C_PANEL.name()}; 
                color: white; 
                border: 2px solid {self.config.C_ACCENT.name()}; 
                padding: 10px; 
                border-radius: 5px; 
                font-weight: bold;
            }}
            QPushButton:hover {{ 
                background-color: {self.config.C_INFO_BG.name()}; 
            }}
            QPushButton:checked {{ 
                background-color: {self.config.C_ACCENT.name()}; 
                color: white; 
            }}
            QTextEdit {{ 
                background-color: {self.config.C_PANEL.name()}; 
                color: #D8DEE9; 
                border: 1px solid {self.config.C_INFO_BG.name()}; 
                font-family: 'Consolas'; 
                font-size: 11px; 
                border-radius: 5px;
                padding: 5px;
            }}
        """)
    
    def setup_ui(self):
        """Build user interface"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # LEFT PANEL
        panel = self.create_control_panel()
        main_layout.addWidget(panel)
        
        # RIGHT PANEL - Map View
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet(
            f"border: 2px solid {self.config.C_ACCENT.name()}; "
            f"background-color: {self.config.C_BG_DARK.name()}; "
            f"border-radius: 5px;"
        )
        main_layout.addWidget(self.view)
        
        # Graphics items
        self.car_item = None
        self.target_items = []
        self.sensor_items = []
        self.trail_item = None
    
    def create_control_panel(self):
        """Create left control panel"""
        panel = QFrame()
        panel.setFixedWidth(300)
        panel.setStyleSheet(f"background-color: {self.config.C_BG_DARK.name()};")
        
        vbox = QVBoxLayout(panel)
        vbox.setSpacing(12)
        
        # Title
        lbl_title = QLabel("🗼 CITY MAP")
        lbl_title.setStyleSheet(
            f"font-weight: bold; font-size: 16px; "
            f"color: {self.config.C_ACCENT.name()}; margin-bottom: 8px;"
        )
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(lbl_title)
        
        # Status
        self.lbl_status = QLabel("Ready to train!\nPress START to begin")
        self.lbl_status.setStyleSheet(
            f"background-color: {self.config.C_INFO_BG.name()}; "
            f"padding: 12px; border-radius: 5px; color: #E5E9F0;"
        )
        self.lbl_status.setWordWrap(True)
        vbox.addWidget(self.lbl_status)
        
        # Buttons
        self.btn_run = QPushButton("▶ START TRAINING")
        self.btn_run.setCheckable(True)
        self.btn_run.clicked.connect(self.toggle_training)
        vbox.addWidget(self.btn_run)
        
        self.btn_reset = QPushButton("↻ RESET")
        self.btn_reset.clicked.connect(self.full_reset)
        vbox.addWidget(self.btn_reset)
        
        vbox.addSpacing(10)
        
        # Reward Chart
        chart_label = QLabel("📊 REWARD HISTORY")
        chart_label.setStyleSheet("font-weight: bold;")
        vbox.addWidget(chart_label)
        
        # Note: Import RewardChart from Part 5
        # self.chart = QWidget()  # Placeholder - use RewardChart from Part 5
        self.chart = RewardChart(
            self.config.C_PANEL,
            self.config.C_ACCENT,
            self.config.C_SUCCESS
        )
        self.chart.setMinimumHeight(150)
        vbox.addWidget(self.chart)
        
        # Statistics
        stats_frame = self.create_stats_panel()
        vbox.addWidget(stats_frame)
        
        # Logs
        log_label = QLabel("📝 TRAINING LOGS")
        log_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        vbox.addWidget(log_label)
        
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        vbox.addWidget(self.log_console)
        
        return panel
    
    def create_stats_panel(self):
        """Create statistics display panel"""
        stats_frame = QFrame()
        stats_frame.setStyleSheet(
            f"background-color: {self.config.C_PANEL.name()}; "
            f"border-radius: 5px; "
            f"border: 1px solid {self.config.C_INFO_BG.name()};"
        )
        
        layout = QGridLayout(stats_frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # Create stat rows
        def add_stat(text, row, color=None):  # ✅ ADD color=None parameter
            label = QLabel(text)
            label.setStyleSheet("color: #88C0D0;")
            value = QLabel("0")
            
            # ✅ ADD this if/else block
            if color:
                value.setStyleSheet(
                    f"color: {color}; "
                    f"font-weight: bold; font-size: 14px;"
                )
            else:
                value.setStyleSheet(
                    f"color: {self.config.C_ACCENT.name()}; "
                    f"font-weight: bold; font-size: 14px;"
                )
            
            layout.addWidget(label, row, 0)
            layout.addWidget(value, row, 1, Qt.AlignmentFlag.AlignRight)
            return value
        
        # # ✅ ADD: Current target and steps at the top
        # self.val_target_current = add_stat("🎯 Target:", 0)
        # self.val_steps = add_stat("📏 Steps:", 1)
        
        # # Add separator
        # separator1 = QLabel("─" * 20)
        # separator1.setStyleSheet("color: #4C566A;")
        # layout.addWidget(separator1, 2, 0, 1, 2)

        self.val_eps = add_stat("Epsilon:", 0)
        self.val_rew = add_stat("Reward:", 1)
        self.val_episode = add_stat("Episode:", 2)
        self.val_success = add_stat("Success:", 3)

        # ✅ ADD: Separator
        separator = QLabel("─" * 20)
        separator.setStyleSheet("color: #4C566A;")
        layout.addWidget(separator, 4, 0, 1, 2)
        
        # ✅ ADD: Target reach counters header
        target_label = QLabel("🎯 TARGET REACH COUNTS")
        target_label.setStyleSheet("color: #88C0D0; font-weight: bold; margin-top: 5px;")
        layout.addWidget(target_label, 5, 0, 1, 2)
        
        # ✅ ADD: Individual target counters with colors
        self.val_target1 = add_stat("Target 1:", 6, self.config.TARGET_COLORS[0].name())
        self.val_target2 = add_stat("Target 2:", 7, self.config.TARGET_COLORS[1].name())
        self.val_target3 = add_stat("Target 3:", 8, self.config.TARGET_COLORS[2].name())
    
        
        return stats_frame
    
    def load_map(self):
        map_path = "simple_grid_map.png"
        meta_path = "map_meta.json"
        
        if not os.path.exists(map_path):
            self.log("❌ Map not found! map_generator.py first")
            QMessageBox.warning(
                self, 
                "Map Missing",
                "Please run 'map_generator.py' first to create the map!"
            )
            sys.exit(1)

        if not os.path.exists(meta_path):
            self.log("❌ map_meta.json not found!")
            QMessageBox.warning(
                self,
                "Metadata Missing",
                "Please run 'map_generator.py' to generate map_meta.json"
            )
            sys.exit(1)

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.map_start = tuple(meta["start"])
        self.map_targets = [tuple(t) for t in meta["targets"]]
        self.PRESET_TARGETS = self.map_targets


        self.log(f"📍 Start loaded at {self.map_start}")
        for i, t in enumerate(self.map_targets, 1):
            self.log(f"🎯 Target {i} loaded at {t}")
        
        self.map_img = QImage(map_path).convertToFormat(QImage.Format.Format_RGB32)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(self.map_img))
        
        # Initialize brain (import CarBrain from Part 4)
        self.brain = CarBrain(self.map_img, self.config)
        # Inject start position from map metadata
        sx, sy = self.map_start
        self.brain.start_pos = QPointF(sx, sy)
        self.brain.car_pos = QPointF(sx, sy)

        
        self.log(f"✅ Map loaded: {map_path}")
    
    def setup_preset_targets(self):
        """Setup 3 preset targets"""
        # Set car start position
        # sx, sy = self.config.START_POS
        sx, sy = self.map_start
        start_pos = QPointF(sx, sy)
        self.brain.set_start_pos(start_pos)
        
        # Create car item (import CarItem from Part 5)
        self.car_item = CarItem(
            self.config.CAR_WIDTH,
            self.config.CAR_HEIGHT,
            self.config.C_ACCENT
        )
        self.scene.addItem(self.car_item)
        self.car_item.setPos(start_pos)
        
        # Create 7 sensor items
        for i in range(7):
            sensor = SensorItem(
                self.config.C_SENSOR_ON,
                self.config.C_SENSOR_OFF
            )
            self.scene.addItem(sensor)
            self.sensor_items.append(sensor)
            
        
        # Add 3 preset targets
        for i, (x, y) in enumerate(self.PRESET_TARGETS):
            pos = QPointF(x, y)
            self.brain.add_target(pos)
            
            # Create target item (import TargetItem from Part 5)
            target = TargetItem(
                self.config.TARGET_COLORS[i],
                self.config.TARGET_LABELS[i],
                is_active=(i == 0)
            )
            target.setPos(pos)
            self.scene.addItem(target)
            self.target_items.append(target)
            
            self.log(f"📍 Target {i+1}: {self.config.TARGET_LABELS[i]} at ({x}, {y})")
        
        # Create path trail
        self.trail_item = PathTrailItem(QColor(233, 69, 96, 80))
        self.scene.addItem(self.trail_item)
        
        self.lbl_status.setText(
            f"✅ Ready to train!\n"
            f"3 targets loaded\n"
            f"Press START"
        )
        self.lbl_status.setStyleSheet(
            f"background-color: {self.config.C_SUCCESS.name()}; "
            f"color: #2E3440; font-weight: bold; "
            f"padding: 12px; border-radius: 5px;"
        )
    
    def toggle_training(self):
        """Start/pause training"""
        if self.btn_run.isChecked():
            self.sim_timer.start(16)  # ~60 FPS
            self.btn_run.setText("⏸ PAUSE")
            self.log("▶ Training started")
        else:
            self.sim_timer.stop()
            self.btn_run.setText("▶ RESUME")
            self.log("⏸ Training paused")
    
    def full_reset(self):
        """Reset everything"""
        self.sim_timer.stop()
        self.btn_run.setChecked(False)
        self.btn_run.setText("▶ START TRAINING")
        
        # Reset brain
        self.brain.reset()
        
        # Clear chart
        self.chart.scores = []
        self.chart.update()
        
        self.log("↻ System reset")
    
    def game_loop(self):
        """
        Main training loop - called every frame
        This is where the magic happens!
        """
        # Get current state
        state, _ = self.brain.get_state()
        
        # Choose action (epsilon-greedy)
        if random.random() < self.brain.epsilon:
            action = random.randint(0, 4)  # Explore
        else:
            with torch.no_grad():
                q = self.brain.policy_net(torch.FloatTensor(state).unsqueeze(0))
                action = q.argmax().item()  # Exploit
        
        # Take step
        # action=1 #to be removed
        next_state, reward, done = self.brain.step(action)
        
        # Store experience
        self.brain.store_experience((state, action, reward, next_state, done))
        
        # Train
        loss = self.brain.optimize()
        self.brain.update_target_network()
        
        # Update visuals
        self.update_visuals()

        # # Update stats
        # # ✅ ADD: Update current target and steps
        # self.val_target_current.setText(f"{self.brain.current_target_idx + 1}/{self.brain.active_targets}")
        # self.val_steps.setText(f"{self.brain.steps_current_target}/{self.brain.max_steps_per_target}")
        
        # Update stats
        self.val_eps.setText(f"{self.brain.epsilon:.3f}")
        self.val_rew.setText(f"{self.brain.score:.0f}")
        self.val_episode.setText(f"{self.brain.episode_count}")
        self.val_success.setText(f"{self.brain.successful_episodes}")

        # ✅ ADD: Update target reach counters
        self.val_target1.setText(f"{self.brain.target_reach_counts[0]}")
        self.val_target2.setText(f"{self.brain.target_reach_counts[1]}")
        self.val_target3.setText(f"{self.brain.target_reach_counts[2]}")
            
        # Handle episode end
        if done:
            self.handle_episode_end()
        
        
    
    def handle_episode_end(self):
        """Handle end of episode"""
        self.brain.finalize_episode(self.brain.score)
        
        targets_reached = self.brain.targets_reached
        
        # ✅ ONLY LOG SIGNIFICANT EVENTS
        if self.brain.alive:
            if targets_reached == 3:
                self.log(f"🏆 Episode {self.brain.episode_count}: ALL TARGETS! Score: {self.brain.score:.0f}")
            # Don't log partial successes unless it's a milestone
        else:
            # Only log crashes every 50 episodes
            if self.brain.episode_count % 50 == 0:
                self.log(f"Episode {self.brain.episode_count}: Crash at T{self.brain.current_target_idx + 1}")
        
        # Log curriculum progress only when close to unlock
        if self.brain.active_targets < 3:
            remaining = (self.brain.target_unlock_threshold * self.brain.active_targets) - self.brain.successful_episodes
            if remaining <= 5:  # Only show when close
                self.log(f"🔜 {remaining} more successes to unlock Target {self.brain.active_targets + 1}")
        
        # Log stats only every 100 episodes
        if self.brain.episode_count % 100 == 0:
            self.log(f"📊 Stats [Ep {self.brain.episode_count}] | "
                    f"Success: {self.brain.successful_episodes} | "
                    f"T1: {self.brain.target_reach_counts[0]} | "
                    f"T2: {self.brain.target_reach_counts[1]} | "
                    f"T3: {self.brain.target_reach_counts[2]}")
        
        self.chart.update_chart(self.brain.score)
        self.brain.reset()
        
        # NO RETURN - This is a UI handler method
            
        
    
    def update_visuals(self):
        """Update all visual elements"""
        # Update car position
        self.car_item.setPos(self.brain.car_pos)
        self.car_item.setRotation(self.brain.car_angle)
        
        # Update sensors
        for i, coord in enumerate(self.brain.sensor_coords):
            self.sensor_items[i].setPos(coord)
            sensor_val = self.brain.get_state()[0][i]
            self.sensor_items[i].set_detecting(sensor_val > 0.5)
        
        # Update targets
        for i, target in enumerate(self.target_items):
            is_active = (i == self.brain.current_target_idx)
            target.set_active(is_active)
        
        # Update path trail
        self.trail_item.set_points(self.brain.path_trail)
        # ✅ ADD: Update status label with detailed info
        self.lbl_status.setText(
            f"🎯 Target: {self.brain.current_target_idx + 1}/{self.brain.active_targets}\n"
            f"📏 Steps: {self.brain.steps_current_target}\n"
            # f"🎲 Epsilon: {self.brain.epsilon:.3f}\n"
            # f"💯 Score: {self.brain.score:.0f}"
    )
   
    
    def log(self, msg):
        """Add message to log console"""
        self.log_console.append(msg)
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key.Key_Space:
            self.btn_run.click()


# ==========================================
# MAIN ENTRY POINT
# ==========================================

def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show window
    window = NeuralNav()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

