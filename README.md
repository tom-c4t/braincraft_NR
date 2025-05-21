
<img src="./Braincraft.png" width="100%">

# Introduction

The computational neuroscience literature abounds with models of individual brain structures, such as the hippocampus, basal ganglia, thalamus, and various cortical areas — from visual to prefrontal. These models typically aim to explain specific functions attributed to each structure. For instance, the basal ganglia are often modeled in the context of decision-making, while the hippocampus is associated with episodic memory and spatial navigation through place cells. However, such models are usually highly abstract and simplified, often relying on a small number of over-engineered neurons and synapses, dealing mostly with abstract inputs and outputs. Consequently, despite decades of work, we still lack an integrated, functional mini-brain — a synthetic neural system capable of performing even simple, continuous embodied tasks in a simulated environment.

The **BrainCraft Challenge** seeks to address this gap by promoting the development of such systems. Specifically, the challenge invites participants to design a biologically inspired, rate-based neural network capable of solving a simple decision task. The network must control an agent (or "bot") situated in a continuous, low-complexity environment. The agent’s sole objective is to locate and reach an energy source in order to maintain its viability. While the task is intentionally simple — as illustrated in Figure 1 — it nonetheless poses a non-trivial challenge for current neuroscience-inspired models because of the hard constraints that have been added (see Methods section). Success will require combining functional neural dynamics with sensorimotor control in a continuous loop, echoing the principles of embodied cognition.

```
┌─────────────────┐
│                 │
│   ┌──┐   ┌──┐   │  ▲ :   Bot start position & orientation (up)
│   │  │   │  │   │  1/2 : Potential energy source location
│ 1 │  │ ▲ │  │ 2 │         (one or the other)
│   │  │   │  │   │
│   └──┘   └──┘   │
│                 │
└─────────────────┘
```

**Figure 1.** **Schematic of the challenge environment.** The bot begins at the center of the arena, facing upward (indicated by the triangle ▲). At each run, the energy source is located at either position 1 or 2 (but not both). The environment is continuous, and the bot moves at a constant speed. The neural model controls only the agent’s steering — i.e., its change in orientation at each time step.

# Methods

## Environment

The environment is a 10x10 square maze with three parallels vertical path as illustrated on Figure 1. The cartesian coordinates inside the maze are normalized such that any position (x,y) ∈ [0,1]×[0,1]. Walls can possess a color c ∈ ℕ⁺ whose semantic is not specified a priori. There exists a single energy source that is located at location 1 or 2 at the start of each run (with equal probability). This location remains constant throughout a run but is unknown to the bot. When the bot goes over a source, its energy level is increased by a specific amount. This lasts until the energy source is depleted, leading eventually to the end of the run by lack of energy. The energy level of a source is also decreased by a specific amount at each time step, independently of the presence of the bot.

## Bot

The simulated bot is circular with a given radius and evolves at a constant speed. The bot can be only controlled on steering. If it hits a wall, its speed remain constant after the hit. The bot has an energy level that is decreased by a given amount after each move or hit. If the energy level of the bot drops to 0, the run is ended.

The bot is equipped with a camera that allows to perceive the environment:

 - **n** distance sensors, spread quasi uniformly between -30° and
   +30° relatively to the heading direction. Each sensor encodes the
   distance to the hit wall
 - **1** bump sensor indicating if the bot has just hit a wall
 - **1** energy gauge indicating the current level of energy
   

**Note**: The framebuffer is strictly equivalent to the combination of
distance and color sensors. It is actually build using only this
information. If you don't need it, you can discard it through the
Win weight matrix.

## Model

The architecture of the model is heavily constrained. It is made of an input that is fed to a pool of neurons that are used to compute the output. Neurons inside the pool are leaky rate neurons with a specific leak rate and activation functions. This is actually very close to an echo state network:

   X(t+1) = (1-λ)•X(t) + λ•f(W•X(t) + Win•I(t)) **(Equation 1)**  
   O(t+1) = Wout•g(X(t+1))                      **(Equation 2)**
 
where:

   - X(t) ∈ ℝⁿ is the state vector
   - I(t) ∈ ℝᴾ is the input vector
   - O(t) ∈ ℝ  is the output
   - W ∈ ℝⁿˣⁿ  is the inner weight matrix
   - Win ∈ ℝᴾˣⁿ is the input weight matrix
   - Wout ∈ ℝⁿˣ¹ is the output weight matrix
   - f and g are two activation functions (usually hyperbolic tangent and identity)
   - λ is the leaking rate

The W matrix can be just anything. From a pure feed-foward network to random recurrent one.  The sole output of the mode lis a relative change in the direction (float) that must be bound to -5° and +5°, relatively to the current heading orientation.

# Evaluation

The evaluation is done in two phases. A time limited training phase where the weights of the model are adjusted and a testing phase where the adjusted weights are used to evaluate the behavior of the model. Given the environment, the optimal strategy is to explore the two potential locations for energy sources and then restrict navigation to the half-loop (going through the central corridor) that includes the enery source. Once a bot has depleted all of its energy, the score of the run is computed. This score corresponds to the total distance made by the bot during the trial. The higher, the better.

## Training phase

The training phase can last at most 100 seconds (real time). During the training phase, the participant is free to use any techniques that are deemed necessary (reinforcement / supervised / unsupervising learning, evolutionary techniques, etc). When the training phase ends, it must returns:

  - W ∈ ℝⁿˣⁿ  (inner weight matrix)
  - Win ∈ ℝᴾˣⁿ (input weight matrix)
  - Wout ∈ ℝⁿˣ¹ (output weight matrix)
  - λ ∈ ℝ or ℝⁿ (global or individual leak rates) 
  - f and g (activation functions)

The training code must be self-sufficient. This means it cannot use external resources (e.g. files) that would have been generated after another (possibly longer) training phase. For the same reason, we recommend to stick to **numpy/scipy/matplotlib dependencies** (only).

**Note**: Any code that has been used during training is not available during testing. This means, for example, that if a participant uses an external reward signal, this signal must be generated by the model itself (e.g. as a derivative of the energy level).

## Testing phase

During the testing phase, the bot is placed at the center of the environment with an upward direction, then a position for the energy is chosen randomly and equations (1) and (2) are iterated until the energy of the bot is fully depleted. The testing phase is made of ten such runs and the final score is the mean of the ten runs.

## Code

To help with the design of a model, the Bot class can be used:

```python
import numpy as np
from bot import Bot

bot = Bot(maze, colormap)
bot.forward(np.radians(+5))
```

The newly created bot has a number of state variables:

 - `maze` (constant, 10x10 array of integers)
 - `colormap` (constant, list of (value, color)
 - `radius` (constant, 0.050)
 - `speed`  (constant, 0.001)
 - `camera` (constant, fov = 60, resolution = 64)
    - `camera.depths` (distances to walls)
    - `camera.values` (color of hit walls)
 - `position` (initially 0.5, 0.5)
 - `direction` (inititally 90°)
 - `energy` (initially 5000)
 - `source` (initially 1000)

These variables can be read (and possibly modified) during training but they won't be accessible during testing (no reading, no writing). To actually move the bot, you need to call the `forward` method. This method will change the direction of the bot, move it forward and update the internal state (camera, hit detection, energy consumption).



