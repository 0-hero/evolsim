import pygame
import pygame.gfxdraw
import sys
import math
import pymunk
import pymunk.pygame_util
import json
import os
import time
import numpy as np

# Gym + RL imports
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

###############################################################
# LOAD CONFIG
###############################################################
CONFIG_FILE = "config.json"
if not os.path.exists(CONFIG_FILE):
    # create a default config if not exists
    default_config = {
        "TRAIN_TIMESTEPS": 2000,
        "EVAL_EVERY": 500,
        "N_ENVS": 4,
        "NET_ARCH": [64,64],
        "MUTATION_RATE": 0.5,
        "DT": 0.02,
        "POPULATION_SIZE":10,
        "SECONDS_PER_GEN":10,
        "ADVANCED_SETTINGS": {
            "grid_enabled": False,
            "grid_size": 1.0,
            "keep_best": True,
            "simulate_in_batches": False,
            "selection_method": "Rank Proportional",
            "recombination_method": "One Point",
            "mutation_method": "Global",
            "mutation_rate":0.5,
            "live_rendering":False
        }
    }
    with open(CONFIG_FILE,"w") as f:
        json.dump(default_config,f)
with open(CONFIG_FILE,"r") as f:
    config = json.load(f)

TRAIN_TIMESTEPS = config["TRAIN_TIMESTEPS"]
EVAL_EVERY = config["EVAL_EVERY"]
N_ENVS = config["N_ENVS"]
NET_ARCH = config["NET_ARCH"]
MUTATION_RATE = config["MUTATION_RATE"]
DT = config["DT"]
POPULATION_SIZE = config["POPULATION_SIZE"]
SECONDS_PER_GEN = config["SECONDS_PER_GEN"]
ADVANCED_SETTINGS = config["ADVANCED_SETTINGS"]

###############################################################
# CONFIG / CONSTANTS
###############################################################
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS = 60

BACKGROUND_COLOR = (230, 230, 230)
BUTTON_BG_COLOR = (220, 220, 220)
BUTTON_ACTIVE_BG = (180, 180, 180)
TEXT_COLOR = (0, 0, 0)
JOINT_COLOR = (200, 0, 0)
BONE_COLOR = (0,0,0)
MUSCLE_COLOR = (200, 0, 0)
GROUND_Y = 700
JOINT_RADIUS = 7

CREATURE_NAME = "Frogger"

###############################################################
# INIT PYGAME
###############################################################
pygame.init()
pygame.display.set_caption("Creature Editor & RL Evolve")
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)
small_font = pygame.font.SysFont("Arial", 14)
large_font = pygame.font.SysFont("Arial", 24)

###############################################################
# APP STATE
###############################################################
MODE = "CREATURE_EDIT"
EDIT_TOOL = "JOINT"
creature_joints = []
creature_bones = []
creature_muscles = []
selected_joint = None
selected_element = None

filename = "creature_save.json"
need_save = False
pending_joint_for_connection = None

edit_num_layers = len(NET_ARCH)
edit_layer_sizes = NET_ARCH[:]

show_network_edit = False
show_advanced_settings = False
show_help = False

training_running = False
training_progress = 0
best_distance = 0.0
current_generation = 1
autoplay = False
simulate_duration = SECONDS_PER_GEN
current_policy = None
training_log = []
final_stats_shown = False
final_stats = {}
current_objective = "rag_doll"  # Default objective

# Add available objectives
AVAILABLE_OBJECTIVES = list(config.get("OBJECTIVES", {}).keys())
if not AVAILABLE_OBJECTIVES:
    AVAILABLE_OBJECTIVES = ["rag_doll"]

###############################################################
# HELPER FUNCTIONS
###############################################################
def save_config():
    config["TRAIN_TIMESTEPS"] = TRAIN_TIMESTEPS
    config["EVAL_EVERY"] = EVAL_EVERY
    config["N_ENVS"] = N_ENVS
    config["NET_ARCH"] = edit_layer_sizes
    config["MUTATION_RATE"] = MUTATION_RATE
    config["DT"] = DT
    config["POPULATION_SIZE"] = POPULATION_SIZE
    config["SECONDS_PER_GEN"] = SECONDS_PER_GEN
    config["ADVANCED_SETTINGS"] = ADVANCED_SETTINGS
    with open(CONFIG_FILE,"w") as f:
        json.dump(config,f)

def draw_text(text, x, y, color=TEXT_COLOR, fnt=font, center=False):
    txt_surf = fnt.render(text, True, color)
    rect = txt_surf.get_rect()
    if center:
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)
    screen.blit(txt_surf, rect)

def point_in_circle(px, py, cx, cy, r):
    return (px - cx)**2 + (py - cy)**2 <= r**2

def save_creature():
    data = {
        "joints": [(j[0], j[1]) for j in creature_joints],
        "bones": creature_bones,
        "muscles": creature_muscles
    }
    with open(filename, "w") as f:
        json.dump(data, f)
    print("Creature saved to", filename)

def load_creature():
    global creature_joints, creature_bones, creature_muscles
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        creature_joints = [(pos[0], pos[1]) for pos in data.get("joints", [])]
        creature_bones = data.get("bones", [])
        creature_muscles = data.get("muscles", [])
        print("Creature loaded from", filename)

def align_creature_on_ground():
    if len(creature_joints)==0:
        return
        
    # Define a better ground position - higher up in the viewport
    VISIBLE_GROUND_Y = WINDOW_HEIGHT - 100  # Place ground 100px from bottom
    
    # Find creature bounds
    min_x = min(x for (x,y) in creature_joints)
    max_x = max(x for (x,y) in creature_joints)
    min_y = min(y for (x,y) in creature_joints)
    max_y = max(y for (x,y) in creature_joints)
    
    # Calculate center position
    center_x = (min_x + max_x) / 2
    creature_width = max_x - min_x
    creature_height = max_y - min_y
    
    # Calculate desired position
    target_x = WINDOW_WIDTH // 2
    target_y = VISIBLE_GROUND_Y - creature_height - JOINT_RADIUS
    
    # Calculate offsets
    x_offset = target_x - center_x
    y_offset = target_y - min_y
    
    # Move all joints
    for i, (x,y) in enumerate(creature_joints):
        new_x = x + x_offset
        new_y = y + y_offset
        creature_joints[i] = (new_x, new_y)

    # Adjust physics ground to match visible ground
    global GROUND_Y
    GROUND_Y = VISIBLE_GROUND_Y

def build_pymunk_space(joints, bones, muscles, objective_type="rag_doll"):
    # Get objective-specific physics parameters
    physics_params = {
        "rag_doll": {
            "gravity": (0.0, 300.0),
            "damping": 0.98,
            "ground_friction": 1.0
        },
        "walking": {
            "gravity": (0.0, 400.0),
            "damping": 0.95,
            "ground_friction": 1.5
        },
        "running": {
            "gravity": (0.0, 400.0),
            "damping": 0.92,
            "ground_friction": 1.2
        },
        "jumping": {
            "gravity": (0.0, 350.0),
            "damping": 0.95,
            "ground_friction": 1.0
        },
        "obstacle_jumping": {
            "gravity": (0.0, 350.0),
            "damping": 0.95,
            "ground_friction": 1.0
        }
    }.get(objective_type, {
        "gravity": (0.0, 300.0),
        "damping": 0.98,
        "ground_friction": 1.0
    })

    space = pymunk.Space()
    space.gravity = physics_params["gravity"]
    space.iterations = 10
    space.damping = physics_params["damping"]
    
    # Ground setup with objective-specific friction
    ground = pymunk.Segment(space.static_body, (-10000, GROUND_Y), (100000, GROUND_Y), 5.0)
    ground.friction = physics_params["ground_friction"]
    ground.elasticity = 0.1
    ground.collision_type = 1
    space.add(ground)
    
    # Add obstacles for obstacle_jumping
    obstacles = []
    if objective_type == "obstacle_jumping":
        try:
            obstacle_config = config["OBJECTIVES"]["obstacle_jumping"]["obstacle_config"]
            initial_x = WINDOW_WIDTH // 2
            
            for i in range(3):
                x = initial_x + (i + 1) * obstacle_config["spacing"]
                obstacle = pymunk.Segment(
                    space.static_body,
                    (x, GROUND_Y - obstacle_config["height"]),
                    (x, GROUND_Y),
                    5.0
                )
                obstacle.friction = 0.5
                obstacle.elasticity = 0.1
                obstacle.collision_type = 4
                space.add(obstacle)
                obstacles.append(obstacle)
                
            def obstacle_collision_handler(arbiter, space, data):
                if not hasattr(space, 'obstacle_collisions'):
                    space.obstacle_collisions = set()
                space.obstacle_collisions.add(arbiter.shapes[1].body)
                return True
                
            obstacle_handler = space.add_collision_handler(4, 2)
            obstacle_handler.begin = obstacle_collision_handler
            
        except (KeyError, TypeError) as e:
            print(f"Warning: Error setting up obstacles: {e}")
    
    # Find bottom-most joints (feet)
    max_y = max(y for _, y in joints)
    min_y = min(y for _, y in joints)
    foot_threshold = min_y + (max_y - min_y) * 0.2
    
    # Create a map of bones connected by muscles
    bones_with_muscles = set()
    muscle_connections = {}  # Maps bone pairs to their connecting muscles
    for i, muscle in enumerate(muscles):
        bones_with_muscles.add(muscle['bone1'])
        bones_with_muscles.add(muscle['bone2'])
        key = (min(muscle['bone1'], muscle['bone2']), max(muscle['bone1'], muscle['bone2']))
        if key not in muscle_connections:
            muscle_connections[key] = []
        muscle_connections[key].append(i)

    # Create joint bodies
    joint_bodies = []
    for i, (x,y) in enumerate(joints):
        is_foot = y <= foot_threshold
        mass = 4.0 if is_foot else 2.0
        moment = pymunk.moment_for_circle(mass, 0, JOINT_RADIUS)
        body = pymunk.Body(mass, moment)
        body.position = (x, y)
        
        shape = pymunk.Circle(body, JOINT_RADIUS)
        shape.friction = 2.0 if is_foot else 0.8
        shape.elasticity = 0.1
        shape.filter = pymunk.ShapeFilter(group=1)
        shape.collision_type = 2
        shape.is_foot = is_foot
        
        space.add(body, shape)
        joint_bodies.append(body)

    # Set up ground collision handler
    if not hasattr(space, 'joint_ground_handler'):
        space.joint_ground_handler = space.add_collision_handler(1, 2)
        space.joint_ground_handler.data["contacts"] = set()
        def begin(arbiter, space, data):
            shape = arbiter.shapes[1]
            if hasattr(shape, 'is_foot') and shape.is_foot:
                body = shape.body
                vx = body.velocity.x
                if abs(vx) > 0.1:
                    force = -vx * body.mass * 50
                    body.apply_force_at_world_point((force, 0), body.position)
            data["contacts"].add(arbiter.shapes[1].body)
            return True
        def separate(arbiter, space, data):
            if arbiter.shapes[1].body in data["contacts"]:
                data["contacts"].remove(arbiter.shapes[1].body)
            return True
        space.joint_ground_handler.begin = begin
        space.joint_ground_handler.separate = separate

    # Create bones
    bone_bodies = []
    motor_constraints = []
    joint_angles = {}
    bone_angles = {}

    for i, (j1, j2) in enumerate(bones):
        p1 = joint_bodies[j1].position
        p2 = joint_bodies[j2].position
        center = ((p1.x + p2.x)/2, (p1.y + p2.y)/2)
        length = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
        angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
        
        density = 1.0
        mass = length * density
        moment = pymunk.moment_for_segment(mass, (-length/2, 0), (length/2, 0), 5)
        bone_body = pymunk.Body(mass, moment)
        bone_body.position = center
        bone_body.angle = angle
        
        bone_shape = pymunk.Segment(bone_body, (-length/2, 0), (length/2, 0), 5)
        bone_shape.friction = 0.8
        bone_shape.elasticity = 0.1
        bone_shape.filter = pymunk.ShapeFilter(group=1)
        bone_shape.collision_type = 3
        
        space.add(bone_body, bone_shape)
        bone_bodies.append(bone_body)
        
        bone_angles[i] = angle
        joint_angles[(j1, i)] = joint_bodies[j1].angle - angle
        joint_angles[(j2, i)] = joint_bodies[j2].angle - angle

        if i in bones_with_muscles:
            # For bones with muscles, use normal pivot joints
            pj1 = pymunk.PivotJoint(joint_bodies[j1], bone_body, p1)
            pj2 = pymunk.PivotJoint(joint_bodies[j2], bone_body, p2)
            space.add(pj1, pj2)

            # Add rotary limits and springs for controlled movement
            if objective_type in ["walking", "running"]:
                max_angle = math.pi/8
            elif objective_type in ["jumping", "obstacle_jumping"]:
                max_angle = math.pi/4
            else:
                max_angle = math.pi/6
                
            if joint_bodies[j1].position.y <= foot_threshold or joint_bodies[j2].position.y <= foot_threshold:
                max_angle *= 0.5

            r1 = pymunk.RotaryLimitJoint(joint_bodies[j1], bone_body, -max_angle, max_angle)
            r2 = pymunk.RotaryLimitJoint(joint_bodies[j2], bone_body, -max_angle, max_angle)
            space.add(r1, r2)

            # Add springs for controlled movement
            if objective_type in ["walking", "running"]:
                spring_stiffness = 35000.0
                spring_damping = 2000.0
            elif objective_type in ["jumping", "obstacle_jumping"]:
                spring_stiffness = 25000.0
                spring_damping = 1000.0
            else:
                spring_stiffness = 30000.0
                spring_damping = 1500.0

            if joint_bodies[j1].position.y <= foot_threshold or joint_bodies[j2].position.y <= foot_threshold:
                spring_stiffness *= 1.5
                spring_damping *= 1.5

            rs1 = pymunk.DampedRotarySpring(joint_bodies[j1], bone_body, joint_angles[(j1, i)], spring_stiffness, spring_damping)
            rs2 = pymunk.DampedRotarySpring(joint_bodies[j2], bone_body, joint_angles[(j2, i)], spring_stiffness, spring_damping)
            space.add(rs1, rs2)

            # Add motors for muscle control
            motor1 = pymunk.SimpleMotor(joint_bodies[j1], bone_body, 0.0)
            motor2 = pymunk.SimpleMotor(joint_bodies[j2], bone_body, 0.0)
            
            base_force = 75000.0 if objective_type in ["walking", "running"] else 50000.0
            foot_force = base_force * 2.0
            
            motor1.max_force = foot_force if joint_bodies[j1].position.y <= foot_threshold else base_force
            motor2.max_force = foot_force if joint_bodies[j2].position.y <= foot_threshold else base_force
            
            space.add(motor1, motor2)
            motor_constraints.extend([(motor1, joint_angles[(j1, i)]), (motor2, joint_angles[(j2, i)])])
        else:
            # For bones without muscles, create rigid connections
            # Use groove joints to maintain exact positions
            groove1 = pymunk.GrooveJoint(joint_bodies[j1], bone_body, (0, 0), (0, 0), (0, 0))
            groove2 = pymunk.GrooveJoint(joint_bodies[j2], bone_body, (0, 0), (0, 0), (0, 0))
            space.add(groove1, groove2)

            # Add pin joints for additional stability
            pin1 = pymunk.PinJoint(joint_bodies[j1], bone_body, (0, 0), (-length/2, 0))
            pin2 = pymunk.PinJoint(joint_bodies[j2], bone_body, (0, 0), (length/2, 0))
            space.add(pin1, pin2)

    # Create muscles
    muscle_structs = []
    muscle_usage = [0.0]*len(muscles)
    for muscle in muscles:
        bone1_body = bone_bodies[muscle['bone1']]
        bone2_body = bone_bodies[muscle['bone2']]
        
        p1 = bone1_body.position
        p2 = bone2_body.position
        init_length = math.sqrt((p2.x-p1.x)**2 + (p2.y-p1.y)**2)
        
        # Calculate initial relative angle between the bones
        init_angle = bone2_body.angle - bone1_body.angle
        
        # Add rotary limit joint to prevent inversion
        rotary_limit = pymunk.RotaryLimitJoint(
            bone1_body, bone2_body,
            init_angle - math.pi/2,
            init_angle + math.pi/2
        )
        space.add(rotary_limit)
        
        # Add damped rotary spring to maintain relative angle
        rotary_spring = pymunk.DampedRotarySpring(
            bone1_body, bone2_body,
            init_angle,
            10000.0,
            800.0
        )
        space.add(rotary_spring)
        
        # Adjust muscle parameters based on objective
        if objective_type in ["walking", "running"]:
            stiffness = 10000.0
            damping = 600.0
        elif objective_type in ["jumping", "obstacle_jumping"]:
            stiffness = 6000.0
            damping = 400.0
        else:
            stiffness = 8000.0
            damping = 500.0
        
        # Create the muscle spring
        spring = pymunk.DampedSpring(
            bone1_body, bone2_body,
            (0, 0),
            (0, 0),
            rest_length=init_length,
            stiffness=stiffness,
            damping=damping
        )
        space.add(spring)
        muscle_structs.append((spring, init_length))
    
    return space, joint_bodies, muscle_structs, muscle_usage, motor_constraints, joint_angles, bone_angles, obstacles

def render_creature(surface, joints, bones, muscles, muscle_usage=None, x_offset=0, alpha=255):
    temp_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    
    # Draw bones first
    for (j1, j2) in bones:
        x1,y1 = joints[j1]
        x2,y2 = joints[j2]
        pygame.draw.line(temp_surf, (*BONE_COLOR, alpha), 
                        (x1+x_offset,y1), (x2+x_offset,y2), 4)
    
    # Draw muscles between bone centers
    if muscle_usage is None:
        muscle_usage = [0]*len(muscles)
        
    for i, muscle in enumerate(muscles):
        # Calculate center points of the connected bones
        j1, j2 = bones[muscle['bone1']]
        x1, y1 = joints[j1]
        x2, y2 = joints[j2]
        center1 = (x1 + x2) / 2, (y1 + y2) / 2
        
        j1, j2 = bones[muscle['bone2']]
        x1, y1 = joints[j1]
        x2, y2 = joints[j2]
        center2 = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Draw muscle line between centers
        thickness = 4 + int(muscle_usage[i]*10)
        pygame.draw.line(temp_surf, (*MUSCLE_COLOR, alpha), 
                        (center1[0]+x_offset, center1[1]), 
                        (center2[0]+x_offset, center2[1]), 
                        thickness)
    
    # Draw joints last (on top)
    for (x,y) in joints:
        pygame.gfxdraw.filled_circle(temp_surf, int(x+x_offset),int(y), 
                                   JOINT_RADIUS, JOINT_COLOR+(alpha,))
        pygame.gfxdraw.aacircle(temp_surf, int(x+x_offset),int(y), 
                               JOINT_RADIUS, JOINT_COLOR+(alpha,))
    
    surface.blit(temp_surf,(0,0))

def in_button(mx,my, bx,by,bw,bh):
    return (mx>=bx and mx<bx+bw and my>=by and my<by+bh)

###############################################################
# UI DRAWING
###############################################################
def draw_top_bar():
    bar_height = 50
    pygame.draw.rect(screen, (245,245,245), (0,0,WINDOW_WIDTH,bar_height))
    draw_text("SIMULATIONS", 10, 15)
    draw_text("CLEAR", 130, 15)
    draw_text(CREATURE_NAME.upper(), 200, 15)
    draw_text("SAVE", 350, 15)
    
    right_section_x = WINDOW_WIDTH - 500
    draw_text("POPULATION", right_section_x, 10)
    draw_text(str(POPULATION_SIZE), right_section_x, 30)
    draw_text("SECONDS PER GEN", right_section_x+120, 10)
    draw_text(str(SECONDS_PER_GEN), right_section_x+120, 30)
    
    # Add objective selector
    draw_text("OBJECTIVE", right_section_x+240, 10)
    draw_text(current_objective.upper(), right_section_x+240, 30)
    # Add navigation arrows
    draw_text("<", right_section_x+240-20, 30)
    draw_text(">", right_section_x+240+len(current_objective)*10+10, 30)
    
    draw_text("RUNNING" if training_running else "STOPPED", right_section_x+400, 15)

def draw_side_buttons():
    buttons = ["JOINT","BONE","MUSCLE","MOVE","DELETE","SELECT"]
    x = 10
    y = 60
    w = 80
    h = 40
    spacing = 50
    for b in buttons:
        color = BUTTON_ACTIVE_BG if EDIT_TOOL==b else BUTTON_BG_COLOR
        pygame.draw.rect(screen, color, (x,y,w,h))
        draw_text(b, x+10,y+10)
        y+=spacing

def draw_main_ui():
    render_creature(screen, creature_joints, creature_bones, creature_muscles)
    pygame.draw.rect(screen, (240,240,240), (0, WINDOW_HEIGHT-60, WINDOW_WIDTH,60))
    draw_text("BACK", 10, WINDOW_HEIGHT-40)
    draw_text("EVOLVE", WINDOW_WIDTH-120, WINDOW_HEIGHT-40)
    draw_text("?", WINDOW_WIDTH-50, 15)
    draw_text("⚙", WINDOW_WIDTH-70, 15)

def draw_network_editor():
    pygame.draw.rect(screen, (20,20,20), (0,0,WINDOW_WIDTH, WINDOW_HEIGHT))
    draw_text("NEURAL NETWORK SETTINGS", WINDOW_WIDTH//2, 30, color=(255,255,255), fnt=large_font, center=True)
    draw_text("Extremely high performance impact!", WINDOW_WIDTH//2, 60, color=(200,200,200), center=True)
    start_x = 100
    start_y = 100
    spacing = 80
    draw_text("NR. OF LAYERS", start_x, start_y, color=(255,255,255))
    draw_text(str(edit_num_layers), start_x+150, start_y, color=(255,255,255))
    draw_text("<", start_x+130, start_y, color=(255,255,255))
    draw_text(">", start_x+170, start_y, color=(255,255,255))
    lx = start_x
    ly = start_y+40
    for i,sz in enumerate(edit_layer_sizes):
        draw_text(str(sz), lx, ly, color=(255,255,255))
        draw_text("<", lx-20, ly, color=(255,255,255))
        draw_text(">", lx+20, ly, color=(255,255,255))
        ly+=spacing
    draw_text("RESET", WINDOW_WIDTH-100, 30, color=(255,255,255))
    draw_text("BACK", 50, WINDOW_HEIGHT-50, color=(255,255,255))

def draw_advanced_settings():
    pygame.draw.rect(screen, (20,20,20), (0,0,WINDOW_WIDTH,WINDOW_HEIGHT))
    gx = 100
    gy = 150
    draw_text("ADVANCED SETTINGS", WINDOW_WIDTH//2, 50, color=(255,255,255), center=True)
    draw_text("GRID", gx, gy, (255,255,255))
    pygame.draw.rect(screen, (255,255,255), (gx+60, gy, 20,20), 2)
    if ADVANCED_SETTINGS["grid_enabled"]:
        pygame.draw.line(screen,(255,255,255),(gx+60,gy),(gx+80,gy+20),2)
        pygame.draw.line(screen,(255,255,255),(gx+80,gy),(gx+60,gy+20),2)
    draw_text("GRID SIZE", gx, gy+40, (255,255,255))
    size_val = ADVANCED_SETTINGS["grid_size"]
    pygame.draw.line(screen,(255,255,255),(gx+140,gy+50),(gx+300,gy+50),2)
    slider_pos = gx+140+int((size_val-0.5)*100)
    pygame.draw.circle(screen,(255,255,255),(slider_pos,gy+50),5)
    draw_text(f"{size_val:.1f}", gx+310, gy+40, (255,255,255))

    draw_text("KEEP BEST CREATURES", gx, gy+80, (255,255,255))
    pygame.draw.rect(screen, (255,255,255), (gx+200, gy+80, 20,20), 2)
    if ADVANCED_SETTINGS["keep_best"]:
        pygame.draw.line(screen,(255,255,255),(gx+200,gy+80),(gx+220,gy+100),2)
        pygame.draw.line(screen,(255,255,255),(gx+220,gy+80),(gx+200,gy+100),2)

    draw_text("SIMULATE IN BATCHES", gx, gy+120, (255,255,255))
    pygame.draw.rect(screen, (255,255,255), (gx+200, gy+120, 20,20), 2)
    if ADVANCED_SETTINGS["simulate_in_batches"]:
        pygame.draw.line(screen,(255,255,255),(gx+200,gy+120),(gx+220,gy+140),2)
        pygame.draw.line(screen,(255,255,255),(gx+220,gy+120),(gx+200,gy+140),2)

    draw_text("SELECTION", WINDOW_WIDTH//2, gy,(255,255,255), center=True)
    draw_text(ADVANCED_SETTINGS["selection_method"].upper(), WINDOW_WIDTH//2, gy+20, (255,255,255), center=True)

    draw_text("RECOMBINATION", WINDOW_WIDTH//2, gy+60,(255,255,255), center=True)
    draw_text(ADVANCED_SETTINGS["recombination_method"].upper(), WINDOW_WIDTH//2, gy+80,(255,255,255), center=True)

    draw_text("MUTATION", WINDOW_WIDTH//2, gy+120,(255,255,255), center=True)
    draw_text(ADVANCED_SETTINGS["mutation_method"].upper(), WINDOW_WIDTH//2, gy+140,(255,255,255), center=True)

    draw_text("MUTATION RATE", WINDOW_WIDTH//2+200, gy,(255,255,255))
    mrate = ADVANCED_SETTINGS["mutation_rate"]
    pygame.draw.line(screen,(255,255,255),(WINDOW_WIDTH//2+200, gy+30),(WINDOW_WIDTH//2+300, gy+30),2)
    mr_pos = WINDOW_WIDTH//2+200+int(mrate*100)-50
    pygame.draw.circle(screen,(255,255,255),(mr_pos,gy+30),5)
    draw_text(f"{int(mrate*100)} %", WINDOW_WIDTH//2+310, gy+20,(255,255,255))

    draw_text("LIVE RENDERING", gx, gy+160, (255,255,255))
    pygame.draw.rect(screen, (255,255,255), (gx+150, gy+160, 20,20), 2)
    if ADVANCED_SETTINGS["live_rendering"]:
        pygame.draw.line(screen,(255,255,255),(gx+150,gy+160),(gx+170,gy+180),2)
        pygame.draw.line(screen,(255,255,255),(gx+170,gy+160),(gx+150,gy+180),2)

    draw_text("CLOSE", gx, WINDOW_HEIGHT-50, (255,255,255))

def draw_training_view():
    pygame.draw.rect(screen, (255,255,255), (0,0,WINDOW_WIDTH,WINDOW_HEIGHT))
    
    # Basic info
    draw_text(f"GENERATION {current_generation}", 50, 50)
    draw_text(f"FITNESS: {best_distance:.1f}%", 50, 70)
    draw_text(f"SIMULATION TIME: {simulate_duration}s", 50, 90)
    
    # Draw objective-specific metrics
    y_offset = 110
    if current_objective == "rag_doll":
        avg_speed = final_stats.get("average_speed", 0.0)
        horiz_distance = final_stats.get("horizontal_distance", 0.0)
        draw_text(f"AVERAGE SPEED: {avg_speed:.2f} m/s", 50, y_offset)
        draw_text(f"HORIZ. DISTANCE: {horiz_distance:.1f}m", 50, y_offset + 20)
    
    elif current_objective in ["walking", "running"]:
        stability = final_stats.get("stability", 0.0)
        energy = final_stats.get("energy_efficiency", 0.0)
        speed = final_stats.get("average_speed", 0.0)
        draw_text(f"STABILITY: {stability:.2f}", 50, y_offset)
        draw_text(f"ENERGY EFFICIENCY: {energy:.2f}", 50, y_offset + 20)
        draw_text(f"SPEED: {speed:.2f} m/s", 50, y_offset + 40)
    
    elif current_objective in ["jumping", "obstacle_jumping"]:
        max_height = final_stats.get("max_height", 0.0)
        landing_stability = final_stats.get("landing_stability", 0.0)
        draw_text(f"MAX HEIGHT: {max_height:.2f}m", 50, y_offset)
        draw_text(f"LANDING STABILITY: {landing_stability:.2f}", 50, y_offset + 20)
        if current_objective == "obstacle_jumping":
            obstacles_cleared = final_stats.get("obstacles_cleared", 0)
            draw_text(f"OBSTACLES CLEARED: {obstacles_cleared}", 50, y_offset + 40)
    
    # Render creature and environment
    render_creature(screen, creature_joints, creature_bones, creature_muscles)
    
    # Draw obstacles for obstacle_jumping
    if current_objective == "obstacle_jumping":
        obstacle_config = config["OBJECTIVES"]["obstacle_jumping"]["obstacle_config"]
        initial_x = WINDOW_WIDTH // 2
        for i in range(3):
            x = initial_x + (i + 1) * obstacle_config["spacing"]
            pygame.draw.line(screen, (100,100,100),
                           (x, GROUND_Y - obstacle_config["height"]),
                           (x, GROUND_Y), 5)
    
    # Bottom controls
    pygame.draw.rect(screen,(240,240,240),(0,WINDOW_HEIGHT-60,WINDOW_WIDTH,60))
    draw_text("BACK", 10, WINDOW_HEIGHT-40)
    draw_text("AUTOPLAY", 100, WINDOW_HEIGHT-40)
    draw_text("DURATION", 200, WINDOW_HEIGHT-40)
    pygame.draw.line(screen,(0,0,0),(270,WINDOW_HEIGHT-30),(370,WINDOW_HEIGHT-30),2)
    pygame.draw.circle(screen,(0,0,0),(270,WINDOW_HEIGHT-30),5)
    draw_text("<", WINDOW_WIDTH-60, WINDOW_HEIGHT-40)
    draw_text(">", WINDOW_WIDTH-40, WINDOW_HEIGHT-40)

    if final_stats_shown:
        # Show final stats overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(220)
        overlay.fill((0,0,0))
        screen.blit(overlay,(0,0))
        draw_text("TRAINING COMPLETE!", WINDOW_WIDTH//2, WINDOW_HEIGHT//2-60,color=(255,255,255),center=True)
        
        # Show objective-specific final stats
        y_offset = WINDOW_HEIGHT//2-20
        for key, value in final_stats.items():
            if key != "fitness" and isinstance(value, (int, float)):
                draw_text(f"{key.replace('_', ' ').title()}: {value:.2f}", 
                         WINDOW_WIDTH//2, y_offset,
                         color=(255,255,255), center=True)
                y_offset += 30
        
        draw_text("Press ESC or click to close", WINDOW_WIDTH//2, 
                 WINDOW_HEIGHT//2+60,color=(255,255,255),center=True)

def draw_help_overlay():
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    overlay.set_alpha(200)
    overlay.fill((0,0,0))
    screen.blit(overlay, (0,0))
    draw_text("HELP SCREEN", WINDOW_WIDTH//2, WINDOW_HEIGHT//2-40, color=(255,255,255), center=True)
    draw_text("This is help info. Click CLOSE to return.", WINDOW_WIDTH//2, WINDOW_HEIGHT//2, color=(255,255,255), center=True)
    draw_text("CLOSE", WINDOW_WIDTH//2-20, WINDOW_HEIGHT//2+40, color=(255,255,255))

###############################################################
# MOUSE & UI INTERACTION
###############################################################
def handle_creature_edit_mouse(mx, my, clicked):
    global EDIT_TOOL, pending_joint_for_connection, selected_joint, selected_element, need_save, show_help, show_advanced_settings, current_objective

    bar_height = 50
    if my < bar_height:
        if clicked:
            if in_button(mx,my,10,0,100,50):
                pass
            elif in_button(mx,my,130,0,50,50):
                clear_creature()
            elif in_button(mx,my,200,0,100,50):
                pass
            elif in_button(mx,my,350,0,50,50):
                save_creature()
                need_save = False
            elif in_button(mx,my,WINDOW_WIDTH-80,0,80,50):
                pass
            elif in_button(mx,my,WINDOW_WIDTH-50,0,20,50):
                # '?'
                show_help = True
            elif in_button(mx,my,WINDOW_WIDTH-70,0,20,50):
                show_advanced_settings = True
            
            # Handle objective selection
            right_section_x = WINDOW_WIDTH - 500
            objective_x = right_section_x + 240
            if in_button(mx,my,objective_x-20,20,20,30):  # Left arrow
                current_idx = AVAILABLE_OBJECTIVES.index(current_objective)
                current_objective = AVAILABLE_OBJECTIVES[(current_idx - 1) % len(AVAILABLE_OBJECTIVES)]
            elif in_button(mx,my,objective_x+len(current_objective)*10+10,20,20,30):  # Right arrow
                current_idx = AVAILABLE_OBJECTIVES.index(current_objective)
                current_objective = AVAILABLE_OBJECTIVES[(current_idx + 1) % len(AVAILABLE_OBJECTIVES)]
        return

    if mx<100 and my>60 and my<60+6*50:
        if clicked:
            btn_idx = (my-60)//50
            tools = ["JOINT","BONE","MUSCLE","MOVE","DELETE","SELECT"]
            if btn_idx<len(tools):
                EDIT_TOOL = tools[btn_idx]
        return

    if my>WINDOW_HEIGHT-60:
        if clicked:
            if in_button(mx,my,10,WINDOW_HEIGHT-60,80,60):
                pass
            elif in_button(mx,my,WINDOW_WIDTH-120,WINDOW_HEIGHT-60,120,60):
                start_training()
        return

    if EDIT_TOOL=="JOINT" and clicked:
        creature_joints.append((mx,my))
        need_save=True
    elif EDIT_TOOL=="BONE" and clicked:
        jindex = find_joint(mx,my)
        if jindex is not None:
            if pending_joint_for_connection is None:
                pending_joint_for_connection = jindex
            else:
                if pending_joint_for_connection != jindex:
                    creature_bones.append((pending_joint_for_connection, jindex))
                    need_save=True
                pending_joint_for_connection=None
    elif EDIT_TOOL=="MUSCLE" and clicked:
        bone_idx = find_closest_bone(mx, my)
        if bone_idx is not None:
            if pending_joint_for_connection is None:
                pending_joint_for_connection = bone_idx
            else:
                if pending_joint_for_connection != bone_idx:
                    creature_muscles.append({
                        'bone1': pending_joint_for_connection,
                        'bone2': bone_idx
                    })
                    need_save = True
                pending_joint_for_connection = None
                
    elif EDIT_TOOL=="MOVE":
        if clicked:
            jindex = find_joint(mx,my)
            if jindex is not None:
                selected_joint = jindex
        else:
            if selected_joint is not None:
                creature_joints[selected_joint] = (mx,my)
                need_save=True
    elif EDIT_TOOL=="DELETE" and clicked:
        jindex = find_joint(mx,my)
        if jindex is not None:
            delete_joint(jindex)
            need_save=True
        else:
            if delete_line(mx,my):
                need_save=True

def bone_exists(a,b):
    if a>b:
        a,b = b,a
    for (x,y) in creature_bones:
        if x>y:
            x,y=y,x
        if x==a and y==b:
            return True
    return False

def find_joint(mx,my):
    for i,(x,y) in enumerate(creature_joints):
        if point_in_circle(mx,my,x,y,JOINT_RADIUS):
            return i
    return None

def find_closest_bone(mx, my, threshold=10):
    closest_bone = None
    min_dist = float('inf')
    
    for i, (j1, j2) in enumerate(creature_bones):
        x1, y1 = creature_joints[j1]
        x2, y2 = creature_joints[j2]
        dist = point_to_line_dist(mx, my, x1, y1, x2, y2)
        if dist < threshold and dist < min_dist:
            min_dist = dist
            closest_bone = i
    
    return closest_bone

def get_attachment_point(mx, my, bone):
    j1, j2 = bone
    x1, y1 = creature_joints[j1]
    x2, y2 = creature_joints[j2]
    
    # Project point onto line
    bone_vec = (x2-x1, y2-y1)
    point_vec = (mx-x1, my-y1)
    bone_len = math.sqrt(bone_vec[0]**2 + bone_vec[1]**2)
    dot_prod = (bone_vec[0]*point_vec[0] + bone_vec[1]*point_vec[1]) / bone_len
    
    # Return percentage along bone (clamped between 0 and 1)
    return max(0.0, min(1.0, dot_prod/bone_len))

def get_bone_attachment_point(mx, my, p1, p2):
    """Calculate where along the bone (0.0 to 1.0) the point (mx,my) projects to"""
    x1, y1 = p1
    x2, y2 = p2
    
    # Vector from p1 to p2 (bone direction)
    bone_vec = (x2-x1, y2-y1)
    # Vector from p1 to click point
    point_vec = (mx-x1, my-y1)
    
    # Calculate dot product and bone length
    bone_length = math.sqrt(bone_vec[0]**2 + bone_vec[1]**2)
    dot_product = (bone_vec[0]*point_vec[0] + bone_vec[1]*point_vec[1])
    
    # Get percentage along bone (clamped between 0 and 1)
    percentage = max(0.0, min(1.0, dot_product/(bone_length**2)))
    return percentage

def delete_joint(jindex):
    global creature_bones, creature_muscles
    creature_joints.pop(jindex)
    new_bones = []
    for (a,b) in creature_bones:
        if a==jindex or b==jindex:
            continue
        na = a if a<jindex else a-1
        nb = b if b<jindex else b-1
        new_bones.append((na,nb))
    creature_bones = new_bones

    new_muscles=[]
    for (a,b) in creature_muscles:
        if a==jindex or b==jindex:
            continue
        na = a if a<jindex else a-1
        nb = b if b<jindex else b-1
        new_muscles.append((na,nb))
    creature_muscles=new_muscles

def delete_line(mx,my):
    threshold = 5
    for i,(a,b) in enumerate(creature_bones):
        x1,y1 = creature_joints[a]
        x2,y2 = creature_joints[b]
        dist = point_to_line_dist(mx,my,x1,y1,x2,y2)
        if dist<threshold:
            creature_bones.pop(i)
            return True
    for i,(a,b) in enumerate(creature_muscles):
        x1,y1 = creature_joints[a]
        x2,y2 = creature_joints[b]
        dist = point_to_line_dist(mx,my,x1,y1,x2,y2)
        if dist<threshold:
            creature_muscles.pop(i)
            return True
    return False

def point_to_line_dist(px,py,x1,y1,x2,y2):
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1
    dot = A*C + B*D
    len_sq = C*C+D*D
    param = -1
    if len_sq!=0:
        param = dot/len_sq
    if param<0:
        xx=x1
        yy=y1
    elif param>1:
        xx=x2
        yy=y2
    else:
        xx=x1+param*C
        yy=y1+param*D
    dx=px-xx
    dy=py-yy
    return math.sqrt(dx*dx+dy*dy)

def clear_creature():
    global creature_joints, creature_bones, creature_muscles
    creature_joints = []
    creature_bones = []
    creature_muscles = []

def handle_advanced_settings_mouse(mx,my,clicked):
    global show_advanced_settings
    if clicked and in_button(mx,my,100,WINDOW_HEIGHT-50,100,50):
        show_advanced_settings=False
        save_config()
    if clicked:
        if in_button(mx,my,160,150,20,20):
            ADVANCED_SETTINGS["grid_enabled"]=not ADVANCED_SETTINGS["grid_enabled"]
        if in_button(mx,my,300,230,20,20):
            ADVANCED_SETTINGS["keep_best"]=not ADVANCED_SETTINGS["keep_best"]
        if in_button(mx,my,300,270,20,20):
            ADVANCED_SETTINGS["simulate_in_batches"]=not ADVANCED_SETTINGS["simulate_in_batches"]
        if in_button(mx,my,250,310,20,20):
            ADVANCED_SETTINGS["live_rendering"]=not ADVANCED_SETTINGS["live_rendering"]

def handle_network_edit_mouse(mx,my,clicked):
    global MODE
    if clicked:
        if in_button(mx,my,50,WINDOW_HEIGHT-50,100,50):
            MODE="CREATURE_EDIT"
        if in_button(mx,my,WINDOW_WIDTH-100,0,100,50):
            reset_network()
            save_config()

def reset_network():
    global edit_layer_sizes, edit_num_layers
    edit_layer_sizes = NET_ARCH[:]
    edit_num_layers = len(edit_layer_sizes)

###############################################################
# TRAINING / RL LOGIC
###############################################################
class CreatureEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, joints, bones, muscles, objective_type="rag_doll", dt=DT):
        super().__init__()
        self.joints_data = joints
        self.bones_data = bones
        self.muscles_data = muscles
        self.dt = dt
        self.objective_type = objective_type
        self.reward_weights = config["OBJECTIVES"][objective_type]["reward_weights"]
        
        # Initialize obstacle data for obstacle_jumping
        self.obstacles = []
        if objective_type == "obstacle_jumping":
            self.obstacle_config = config["OBJECTIVES"]["obstacle_jumping"]["obstacle_config"]
        
        self.space = None
        self.body_list = None
        self.muscle_structs = None
        self.muscle_usage = None
        self.motor_constraints = None
        self.joint_angles = None
        self.bone_angles = None
        
        # Enhanced observation space
        # Base observations: For each joint: [x, y, vx, vy, angle, angular_vel]
        # For each bone: [angle, angular_vel]
        # For each muscle: [current_length/rest_length]
        # Plus ground contacts
        joint_dim = len(joints) * 6
        bone_dim = len(bones) * 2
        muscle_dim = len(muscles)
        obs_dim = joint_dim + bone_dim + muscle_dim + len(joints)  # +len(joints) for ground contacts
        
        # Add objective-specific observations
        if objective_type == "obstacle_jumping":
            obs_dim += 2  # Distance to next obstacle, height of obstacle
            
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        # Action space includes both muscle activation and joint motor control
        self.action_space = spaces.Box(-1, 1, shape=(len(muscles) + len(bones)*2,), dtype=np.float32)
        self.max_steps = 50*SECONDS_PER_GEN
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.space:
            for c in self.space.constraints:
                self.space.remove(c)
            for s in self.space.shapes:
                self.space.remove(s)
            for b in self.space.bodies:
                self.space.remove(b)
        
        # Build physics space with objective-specific elements
        (
            self.space,
            self.body_list,
            self.muscle_structs,
            self.muscle_usage,
            self.motor_constraints,
            self.joint_angles,
            self.bone_angles,
            self.obstacles  # Now properly receiving obstacles
        ) = build_pymunk_space(
            self.joints_data,
            self.bones_data,
            self.muscles_data,
            self.objective_type
        )
        
        self.steps = 0
        self.initial_x = np.mean([b.position.x for b in self.body_list])
        self.initial_y = np.mean([b.position.y for b in self.body_list])
        self.max_height = self.initial_y
        self.last_positions = [(b.position.x, b.position.y) for b in self.body_list]
        self.obstacle_collisions = set()  # Track obstacle collisions
        
        obs = self._get_obs()
        info = {}
        return obs, info

    def _calculate_stability(self):
        # Calculate stability based on joint movement and orientation
        stability = 0.0
        current_positions = [(b.position.x, b.position.y) for b in self.body_list]
        
        # Penalize excessive joint movement
        movement = sum(math.sqrt((x2-x1)**2 + (y2-y1)**2) 
                      for (x1,y1), (x2,y2) in zip(self.last_positions, current_positions))
        stability -= movement * 0.01
        
        # Penalize non-upright orientation of bones
        for j1, j2 in self.bones_data:
            x1, y1 = self.body_list[j1].position
            x2, y2 = self.body_list[j2].position
            angle = abs(math.atan2(y2-y1, x2-x1))
        stability -= abs(angle - math.pi/2) * 0.1  # Penalize deviation from vertical
            
        self.last_positions = current_positions
        return max(0.0, 1.0 + stability)  # Normalize to [0,1]

    def _calculate_energy_efficiency(self):
        # Calculate energy efficiency based on muscle usage
        total_usage = sum(abs(u) for u in self.muscle_usage)
        return 1.0 - min(1.0, total_usage / len(self.muscle_usage))

    def step(self, action):
        # Split action into muscle and motor controls
        muscle_actions = action[:len(self.muscle_structs)]
        motor_actions = action[len(self.muscle_structs):]
        
        # Apply muscle actions
        for i, ((spring, base_len), act) in enumerate(zip(self.muscle_structs, muscle_actions)):
            self.muscle_usage[i] = 0.8*self.muscle_usage[i] + 0.2*abs(act)
            new_len = base_len * (1.0 + act * 0.2)
            spring.rest_length = new_len
        
        # Apply motor actions
        for (motor, rest_angle), act in zip(self.motor_constraints, motor_actions):
            target_rate = act * 5.0
            motor.rate = target_rate
        
        # Update obstacle collisions before step
        if hasattr(self.space, 'obstacle_collisions'):
            self.obstacle_collisions = self.space.obstacle_collisions.copy()
        
        self.space.step(self.dt)
        self.steps += 1
        
        # Update max height for jumping objectives
        current_y = np.mean([b.position.y for b in self.body_list])
        self.max_height = min(self.max_height, current_y)  # Note: y increases downward
        
        obs = self._get_obs()
        
        # Calculate reward based on objective
        reward = 0.0
        avg_x = np.mean([b.position.x for b in self.body_list])
        forward_movement = (avg_x - self.initial_x) / 100.0
        
        if self.objective_type == "rag_doll":
            reward = forward_movement
            
        elif self.objective_type in ["walking", "running"]:
            stability = self._calculate_stability()
            energy_efficiency = self._calculate_energy_efficiency()
            speed = forward_movement / self.steps if self.steps > 0 else 0
            
            reward = (
                self.reward_weights["forward_movement"] * forward_movement +
                self.reward_weights.get("stability", 0) * stability +
                self.reward_weights.get("speed", 0) * speed +
                self.reward_weights.get("energy_efficiency", 0) * energy_efficiency
            )
            
        elif self.objective_type == "jumping":
            height_reward = (self.initial_y - self.max_height) / 100.0
            landing_stability = self._calculate_stability()
            
            reward = (
                self.reward_weights["max_height"] * height_reward +
                self.reward_weights["forward_movement"] * forward_movement +
                self.reward_weights["landing_stability"] * landing_stability
            )
            
        elif self.objective_type == "obstacle_jumping":
            height_reward = (self.initial_y - self.max_height) / 100.0
            landing_stability = self._calculate_stability()
            
            # Check obstacle clearance
            obstacle_reward = 0.0
            if len(self.obstacles) > 0:
                # Cache the current position
                current_pos = avg_x
                # Find nearest obstacle ahead
                nearest_obstacle = min(
                    (obs for obs in self.obstacles if obs.a.x > current_pos),
                    key=lambda obs: obs.a.x - current_pos,
                    default=None
                )
                
                if nearest_obstacle:
                    if current_pos > nearest_obstacle.a.x:  # Passed obstacle
                        if not self.obstacle_collisions:  # No collision with this obstacle
                            obstacle_reward = 1.0
                        else:
                            obstacle_reward = 0.2  # Partial reward for passing with collision
            
            reward = (
                self.reward_weights["obstacle_clearance"] * obstacle_reward +
                self.reward_weights["forward_movement"] * forward_movement +
                self.reward_weights["landing_stability"] * landing_stability
            )
        
        # Check termination conditions
        done = False
        
        # Time-based termination
        if self.steps >= self.max_steps:
            done = True
            
        # Objective-specific termination conditions
        if self.objective_type in ["walking", "running"]:
            # Check if creature has fallen (center of mass too low)
            avg_y = np.mean([b.position.y for b in self.body_list])
            if avg_y > GROUND_Y - 20:  # Too close to ground
                done = True
                reward -= 1.0  # Penalty for falling
                
        elif self.objective_type in ["jumping", "obstacle_jumping"]:
            # Check for hard landings or crashes
            if len(self.space.joint_ground_handler.data["contacts"]) > len(self.joints_data) * 0.5:
                # More than half the joints touching ground - crash landing
                done = True
                reward -= 1.0  # Penalty for crash
                
            # Check for obstacle collisions in obstacle_jumping
            if self.objective_type == "obstacle_jumping" and self.obstacle_collisions:
                reward -= 0.5  # Penalty for hitting obstacle
        
        info = {}
        return obs, reward, done, False, info

    def _get_obs(self):
        obs = []
        # Joint observations
        for b in self.body_list:
            obs.extend([
                b.position.x,
                b.position.y,
                b.velocity.x,
                b.velocity.y,
                b.angle,
                b.angular_velocity
            ])
        
        # Bone angles and angular velocities
        for j1, j2 in self.bones_data:
            angle = math.atan2(
                self.body_list[j2].position.y - self.body_list[j1].position.y,
                self.body_list[j2].position.x - self.body_list[j1].position.x
            )
            angular_vel = (self.body_list[j2].angular_velocity + self.body_list[j1].angular_velocity) / 2
            obs.extend([angle, angular_vel])
        
        # Muscle length ratios
        for spring, rest_length in self.muscle_structs:
            p1 = spring.a.position
            p2 = spring.b.position
            current_length = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            obs.append(current_length / rest_length)
        
        # Ground contacts
        for body in self.body_list:
            obs.append(1.0 if body in self.space.joint_ground_handler.data["contacts"] else 0.0)
        
        # Add objective-specific observations
        if self.objective_type == "obstacle_jumping":
            avg_x = np.mean([b.position.x for b in self.body_list])
            nearest_obstacle = min(
                (obs for obs in self.obstacles if obs.a.x > avg_x),
                key=lambda obs: obs.a.x - avg_x,
                default=None
            )
            if nearest_obstacle:
                obs.extend([
                    nearest_obstacle.a.x - avg_x,  # Distance to next obstacle
                    nearest_obstacle.a.y - GROUND_Y  # Height of obstacle
                ])
            else:
                obs.extend([1000.0, 0.0])  # No more obstacles
        
        return np.array(obs, dtype=np.float32)

    def get_render_data(self):
        jpositions = [(b.position.x, b.position.y) for b in self.body_list]
        return jpositions, self.bones_data, self.muscles_data, self.muscle_usage

    def get_final_metrics(self):
        final_x = np.mean([b.position.x for b in self.body_list])
        horizontal_distance = final_x - self.initial_x
        simulation_time = self.steps * self.dt
        average_speed = horizontal_distance / simulation_time if simulation_time > 0 else 0
        
        metrics = {
            "horizontal_distance": horizontal_distance,
            "simulation_time": simulation_time,
            "average_speed": average_speed,
        }
        
        # Add objective-specific metrics
        if self.objective_type in ["walking", "running"]:
            metrics.update({
                "stability": self._calculate_stability(),
                "energy_efficiency": self._calculate_energy_efficiency()
            })
        elif self.objective_type in ["jumping", "obstacle_jumping"]:
            metrics.update({
                "max_height": self.initial_y - self.max_height,
                "landing_stability": self._calculate_stability()
            })
            if self.objective_type == "obstacle_jumping":
                metrics["obstacles_cleared"] = sum(
                    1 for obs in self.obstacles
                    if final_x > obs.a.x
                )
        
        # Set fitness based on objective
        if self.objective_type == "rag_doll":
            metrics["fitness"] = horizontal_distance
        else:
            # Use the same reward calculation as in step()
            _, reward, _, _, _ = self.step(np.zeros(self.action_space.shape))
            metrics["fitness"] = reward
        
        return metrics

    def close(self):
        pass

def make_env():
    return CreatureEnv(creature_joints, creature_bones, creature_muscles)

def start_training():
    global MODE, training_running, current_policy, best_distance, current_generation, final_stats_shown
    align_creature_on_ground()  # ensure creature stands on the ground
    MODE="TRAINING_VIEW"
    training_running=True
    best_distance = 0
    current_generation = 1
    final_stats_shown = False

    def env_fn():
        return CreatureEnv(
            creature_joints, 
            creature_bones, 
            creature_muscles, 
            objective_type=current_objective,
            dt=DT
        )

    env = DummyVecEnv([env_fn for _ in range(N_ENVS)])

    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import MlpExtractor

    class CustomActorCriticPolicy(ActorCriticPolicy):
        def _build_mlp_extractor(self):
            policy_layers = []
            value_layers = []
            last_dim_policy = self.features_dim
            last_dim_value = self.features_dim
            for sz in NET_ARCH:
                policy_layers.append(nn.Linear(last_dim_policy, sz))
                policy_layers.append(nn.ReLU())
                last_dim_policy=sz
                value_layers.append(nn.Linear(last_dim_value,sz))
                value_layers.append(nn.ReLU())
                last_dim_value=sz
            self.mlp_extractor = MlpExtractor(
                feature_dim=self.features_dim,
                net_arch=[],
                activation_fn=nn.ReLU,
                device=self.device
            )
            self.mlp_extractor.policy_net = nn.Sequential(*policy_layers)
            self.mlp_extractor.value_net = nn.Sequential(*value_layers)
            self.mlp_extractor.latent_dim_pi = last_dim_policy
            self.mlp_extractor.latent_dim_vf = last_dim_value

    model = PPO(policy=CustomActorCriticPolicy, env=env, verbose=0, n_steps=512, batch_size=64, n_epochs=5, learning_rate=3e-4)

    steps_done = 0
    global training_progress, final_stats

    while steps_done<TRAIN_TIMESTEPS:
        model.learn(total_timesteps=EVAL_EVERY, reset_num_timesteps=False)
        steps_done+=EVAL_EVERY
        metrics = evaluate_policy(env, model)
        # fitness is now based on objective
        if metrics["fitness"]>best_distance:
            best_distance = metrics["fitness"]
        current_generation +=1
        training_progress = steps_done/TRAIN_TIMESTEPS

        # store stats for display
        final_stats = metrics

        if ADVANCED_SETTINGS["live_rendering"]:
            rollout_view(env, model) 
        pygame.event.pump()
        screen.fill((255,255,255))
        draw_training_view()
        pygame.display.flip()
    
    current_policy = model
    training_running=False
    # After training done, show final stats and pause
    final_stats_shown = True
    while True:
        mx,my = pygame.mouse.get_pos()
        clicked=False
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                sys.exit()
            if event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE:
                return
            if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                return
        screen.fill((255,255,255))
        draw_training_view()
        pygame.display.flip()
        clock.tick(FPS)

def evaluate_policy(env,model):
    # Evaluate using the first environment (env.envs[0])
    single_env = env.envs[0]
    obs,_ = single_env.reset()
    done=False
    steps=0
    while not done and steps<single_env.max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = single_env.step(action)
        steps+=1
    # get final metrics
    metrics = single_env.get_final_metrics()
    return metrics

def rollout_view(env, model, final=False):
    obs = env.reset()
    done = [False]*env.num_envs
    steps = 0
    max_steps = env.envs[0].max_steps
    
    # Calculate grid layout
    num_envs = env.num_envs
    num_cols = int(math.sqrt(num_envs))  # number of columns
    num_rows = math.ceil(num_envs / num_cols)  # number of rows
    
    # Calculate cell size for each environment view
    cell_width = WINDOW_WIDTH // num_cols
    cell_height = WINDOW_HEIGHT // num_rows
    
    # Calculate margin for creature within cell
    margin = 50  # pixels from bottom of cell
    
    while not all(done) and steps<max_steps:
        pygame.event.pump()
        screen.fill((255,255,255))
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, truncs = env.step(actions)
        
        # Draw each env in its grid cell
        for i, env_ in enumerate(env.envs):
            # Calculate grid position
            row = i // num_cols
            col = i % num_cols
            
            # Get render data
            jpos, bones, muscles, musage = env_.get_render_data()
            
            # Find the creature's height to position it properly
            min_y = min(y for (x,y) in jpos)
            max_y = max(y for (x,y) in jpos)
            creature_height = max_y - min_y
            
            # Calculate base y position for this cell
            cell_base_y = (row + 1) * cell_height - margin  # Bottom of cell minus margin
            
            # Calculate position adjustments
            x_offset = col * cell_width + (cell_width // 2)
            y_offset = cell_base_y - GROUND_Y  # Adjust ground level for this cell
            
            # Adjust joint positions for this cell
            adjusted_jpos = []
            for (x,y) in jpos:
                # Center within cell and maintain relative positions
                adj_x = x - (WINDOW_WIDTH//2) + x_offset
                adj_y = y + y_offset
                adjusted_jpos.append((adj_x, adj_y))
            
            # Draw cell border
            pygame.draw.rect(screen, (200,200,200), 
                           (col*cell_width, row*cell_height, cell_width, cell_height), 1)
            
            # Draw ground line for this cell
            ground_y = cell_base_y
            pygame.draw.line(screen, (150,150,150),
                           (col*cell_width, ground_y),
                           ((col+1)*cell_width, ground_y), 2)
            
            # Render creature in its cell
            render_creature(screen, adjusted_jpos, bones, muscles, 
                          muscle_usage=musage, alpha=255)
        
        draw_text("LIVE RENDERING", 10,10)
        draw_text("Press ESC to quit viewing",10,30)
        pygame.display.flip()
        clock.tick(30)
        steps+=1
        
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                sys.exit()
            if event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE:
                return
        done = dones
    
    if final:
        time.sleep(2)

###############################################################
# MAIN LOOP
###############################################################
load_creature()

running=True
selected_joint=None
while running:
    mx,my = pygame.mouse.get_pos()
    clicked=False
    events = pygame.event.get()
    for event in events:
        if event.type==pygame.QUIT:
            running=False
        if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
            clicked=True
        if event.type==pygame.MOUSEBUTTONUP and event.button==1:
            if EDIT_TOOL=="MOVE":
                selected_joint=None
        if event.type==pygame.KEYDOWN:
            if event.key==pygame.K_n:
                MODE="NETWORK_EDIT"
            if event.key==pygame.K_ESCAPE and show_help:
                show_help=False

    screen.fill(BACKGROUND_COLOR)

    if MODE=="CREATURE_EDIT":
        if not show_help and not show_advanced_settings:
            handle_creature_edit_mouse(mx,my, clicked)
        draw_top_bar()
        draw_side_buttons()
        draw_main_ui()
        if show_help:
            draw_help_overlay()
            if clicked and in_button(mx,my, WINDOW_WIDTH//2-20, WINDOW_HEIGHT//2+40,80,30):
                show_help=False
        if show_advanced_settings:
            MODE="ADVANCED_SETTINGS"
            show_advanced_settings=False

    elif MODE=="NETWORK_EDIT":
        handle_network_edit_mouse(mx,my, clicked)
        draw_network_editor()

    elif MODE=="ADVANCED_SETTINGS":
        handle_advanced_settings_mouse(mx,my,clicked)
        draw_advanced_settings()

    elif MODE=="TRAINING_VIEW":
        if clicked and final_stats_shown:
            # if final stats are shown, close on click
            final_stats_shown=False
        if clicked:
            if in_button(mx,my,10,WINDOW_HEIGHT-60,80,60):
                MODE="CREATURE_EDIT"
        draw_training_view()

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
