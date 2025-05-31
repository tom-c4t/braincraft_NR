
<img src="./data/braincraft.png" width="100%">

### Table of Contents  
- [Introduction](#introduction) ....................... *Rationale for the challenge*  
- [Methods](#methods) ............................. *Challenge rules*  
- [Evaluation](#evaluation) .......................... *Scoring system*  
- [Discussion](#discussion) ......................... *Frequently asked questions*  
- [Results](#results) ............................... *Leader board*  

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
         distance to the wall that has been hit.
 - **1** bump sensor indicating if the bot has just hit a wall
 - **1** energy gauge indicating the current level of energy
 - **1** constant value of 1 (might be used for bias)
   

**Note**: The framebuffer is strictly equivalent to the combination of
distance and color sensors. It is actually build using only this
information. If you don't need it, you can discard it through the
Win weight matrix.

## Model

The architecture of the model is subject to strict constraints. It consists of an input layer connected to a reservoir (or pool) of neurons, which in turn is used to compute the output. The neurons within the reservoir are leaky rate units characterized by a specific leak rate and activation functions. This structure closely resembles that of an Echo State Network (ESN), and is governed by the following equations:

- **Equation `1`:** X(t+1) = (1-λ)•X(t) + λ•f(W•X(t) + Win•I(t))
- **Equation `2`:** O(t+1) = Wout•g(X(t+1))

where:

   - X(t) ∈ ℝⁿ is the state vector
   - I(t) ∈ ℝᴾ is the input vector
   - O(t) ∈ ℝ  is the output
   - W ∈ ℝⁿˣⁿ  is the recurrent (or inner) weight matrix
   - Win ∈ ℝᴾˣⁿ is the input weight matrix
   - Wout ∈ ℝⁿˣ¹ is the output weight matrix
   - f and g are activation functions (typically hyperbolic tangent and identity, respectively)
   - λ is the leaking rate

The inner weight matrix W can be arbitrarily defined, ranging from purely feedforward structures to random recurrent topologies. The model produces a single scalar output, representing a relative change in heading direction. This output must be constrained to lie within the range [–5°, +5°] relative to the agent’s current orientation.


# Evaluation

The evaluation procedure comprises two distinct phases: a time-constrained training phase, during which the model parameters are optimized, followed by a testing phase, wherein the learned parameters are employed to assess the model’s performance. Within the given environment, the optimal strategy involves initially exploring both potential energy source locations, subsequently restricting navigation to the half-loop—traversing the central corridor—that contains the identified energy source. Upon complete depletion of the agent’s energy reserves, the trial concludes, and a performance score is computed. This score reflects the total distance traversed by the agent during the trial; higher values indicate superior performance.

## Training phase

The training phase is limited to a maximum duration of 100 seconds of user time. Within this phase, participants are free to employ any learning paradigm deemed appropriate, including reinforcement learning, supervised or unsupervised learning, and evolutionary algorithms. Upon completion of the training phase, the following elements must be returned:

  - `W` ∈ ℝⁿˣⁿ  (inner weight matrix)
  - `Win` ∈ ℝᴾˣⁿ (input weight matrix)
  - `Wout` ∈ ℝⁿˣ¹ (output weight matrix)
  - `λ` ∈ ℝ or ℝⁿ (global or individual leak rates) 
  - `f` and `g` (activation functions)
  - `warmup` duration (bot don't move before warmup period is over)

The training code must be fully self-contained. In particular, it must not rely on any external resources or data (e.g., files or models) that may have been generated during a prior or extended training phase. To ensure reproducibility and compatibility, it is strongly recommended that submissions restrict their dependencies to [NumPy](https://numpy.org), [SciPy](https://scipy.org), and [Matplotlib](https://matplotlib.org) only.

**Note**: Any code utilized during the training phase is not available during the testing phase. Consequently, if, for example, a participant employs an external reward signal during training, this signal must be internally generated by the model itself — e.g., derived from the agent’s own state variables such as the rate of change of the energy level.

## Testing phase

During the testing phase, the agent is initialized at the center of the environment, oriented upward. An energy source location is then selected at random, and Equations (1) and (2) are iteratively applied until the agent’s energy is fully depleted. The testing phase comprises ten such trials, and the final performance score is computed as the mean distance traveled across all ten runs.


```Python
from challenge import train, evaluate

def model():
    # do something and returns your model
    # you can send intermediate models (yield) as well.
    # See the player_random.py example

    ...
    
    yield default_model
    while True:
        ...
        evaluate(better_model, runs=3)
        yield better_model
    return best_model

evaluate(train(model, timeout=100))
```

## Code helpers

To help with the design and debug of a model, you can use the [Bot](./bot.py) class that offer
a number of state variables:

 - `radius` (constant, 0.05)
 - `speed` (constant, 0.01)
 - `camera` (constant, fov = 60, resolution = 64)
    - `camera.depths` (wall distances, n=resolution=64)
    - `camera.values` (walls color, n=resolution=64)
 - `position` (initially 0.5, 0.5)
 - `direction` (inititally 90° ± 5°)
 - `energy` (initially 1000)
 - `source` (initially 1000)

These variables can be read (and possibly modified) during training but they won't be accessible during testing (no reading, no writing). To actually move the bot, you need to call the `forward` method. This method will change the direction of the bot, move it forward and update the internal state (camera, hit detection, energy consumption). The evaluation method has also a debug flag that may be helpful to visualize the behavior of your model (see Figure 2).

![](./data/debug.png)

**Figure 2.** **Debug view during evaluation.** The left part is a bird-eye view of the environment where the yellow part is the unique source of energy. The right part is a first-person view build from the set of 64 sensors that is not needed during evaluation (but it might  help debug).


# Discussion

- **Why 1000 neurons?** Since [It Takes Two Neurons To Ride a
  Bicycle](https://paradise.caltech.edu/~cook/papers/TwoNeurons.pdf),
  I decided that 1000 neurons should be more than enough to solve such a
  *simple* task.

- **Why 100 seconds?** Because I want any student to be able to enter
  the challenge without having access to a cluster: a laptop CPU
  should be just fine. Note that it is also much better for Earth

- **Why 10 runs?** Having only 2 possible environments, 10 runs should
  be enough to give a fair account of your model performance.

- **Why 2 choices?** If you consider the abstraction of the task,
  there is really two choices at branching points.
 
- **Why no reward?** Because it is easy to generate your own reward
  signal from the derivation of energy. It it?

- **Can I run my code on a supercomputer?** Yes, but the official
  evaluation will be done on my machine which is a M1 Macbook pro.

- **Can I change the rules?** Of course no.

- **May I propose some rule changes then?** Sure. Just open an issue with
  the reason you want to change this or that rule.

- **I've found a bug in the simulator** Open an issue and report it,
  with a fix if possible.

# Results

Here is the current leader board. If you think you can do better, send me a link to your code. I'll evaluate it and add a line below.


👤 Author     |  🗓️  Date      | 📄 File        | 📊  Score
---------- | ---------- | ----------- | ---------------
[@rougier] | 01/06/2025 | [random.py] | **0.62** ± 0.37




[@rougier]: https://github.com/rougier
[random.py]: ./braincraft/player_random.py
