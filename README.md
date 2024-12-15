# Evolution Simulator

A physics-based creature evolution simulator where you can design creatures with joints, bones, and muscles, then train them using reinforcement learning to perform various tasks like walking, running, jumping, and obstacle clearing.

## Features

### Creature Editor
- **Joint Creation**: Add joints (red dots) that serve as connection points
- **Bone Creation**: Connect joints with rigid bones (black lines)
- **Muscle Creation**: Add muscles (red lines) between bones that can contract and expand
- **Edit Tools**: Move, delete, and modify creature parts
- **Save/Load**: Save and load your creature designs

### Training Objectives
- **Rag Doll**: Basic physics simulation without specific goals
- **Walking**: Train the creature to walk efficiently
- **Running**: Optimize for speed and stability
- **Jumping**: Train for maximum jump height and distance
- **Obstacle Jumping**: Train to clear obstacles while maintaining stability

### Physics Features
- Realistic joint constraints and limits
- Muscle contraction and expansion
- Ground friction and contact handling
- Energy efficiency tracking
- Stability calculations

### Training Features
- Reinforcement Learning using Stable Baselines 3 (PPO)
- Multiple parallel training environments
- Live training visualization
- Customizable neural network architecture
- Advanced training settings

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd evolsim
```

2. Create a virtual environment and activate it:
```bash
conda create -n evolution python=3.10
conda activate evolution
```

3. Install required packages:
```bash
pip install pygame pymunk gymnasium stable-baselines3 torch numpy
```

## Usage

1. Run the simulator:
```bash
python main.py
```

2. Create a creature:
   - Click "JOINT" to add joints
   - Select "BONE" to connect joints
   - Use "MUSCLE" to add muscles between bones
   - Save your creation

3. Configure training:
   - Select an objective (walking, running, etc.)
   - Adjust population size and generation duration
   - Modify neural network architecture if needed
   - Configure advanced settings

4. Start evolution:
   - Click "EVOLVE" to begin training
   - Watch the live visualization
   - Monitor progress metrics

## Configuration

The `config.json` file contains various settings:

```json
{
    "TRAIN_TIMESTEPS": 2000,
    "EVAL_EVERY": 500,
    "N_ENVS": 4,
    "NET_ARCH": [64,64],
    "MUTATION_RATE": 0.5,
    "DT": 0.02,
    "POPULATION_SIZE": 10,
    "SECONDS_PER_GEN": 10,
    "ADVANCED_SETTINGS": {
        "grid_enabled": false,
        "grid_size": 1.0,
        "keep_best": true,
        "simulate_in_batches": false,
        "selection_method": "Rank Proportional",
        "recombination_method": "One Point",
        "mutation_method": "Global",
        "mutation_rate": 0.5,
        "live_rendering": false
    }
}
```

## Controls

### Editor Mode
- Left Click: Use selected tool
- N: Open neural network settings
- ESC: Close overlays
- ?: Show help

### Training Mode
- ESC/Click: Return to editor
- Autoplay: Toggle continuous simulation
- Duration: Adjust simulation length

## Tips for Creating Successful Creatures

1. **Joint Placement**:
   - Place joints symmetrically for better balance
   - Keep foot joints slightly wider than upper joints
   - Don't make the creature too tall or unstable

2. **Bone Structure**:
   - Create a strong central structure
   - Use shorter bones for better stability
   - Ensure good support for the creature's weight

3. **Muscle Configuration**:
   - Add muscles in pairs for balanced movement
   - Don't over-muscle the creature
   - Consider the direction of intended movement

4. **Training**:
   - Start with simpler objectives (walking before running)
   - Use larger population sizes for better exploration
   - Increase training duration for complex behaviors
   - Monitor energy efficiency and stability metrics

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by various evolution simulators and genetic algorithms
- Built using PyGame, Pymunk, and Stable Baselines 3
- Special thanks to the open-source community 
