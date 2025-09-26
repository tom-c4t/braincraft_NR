# import python packages
import numpy as np
import actor_net
import critic_net
from ReplayBuffer import ReplayBuffer
from bot import Bot
from environment_1 import Environment

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 40
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.003
# Discount factor 
GAMMA = 0.99
# Soft target update param
TAU = 0.001
# Parameters for neural net
HIDDEN1_UNITS = 1000
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64
# upper bound on action
ACTION_BOUND=5
      
    
n = Bot.camera.resolution 
state_dim = n + 3
action_dim = 1
action_bound = ACTION_BOUND

# Create actor and critic nets
actor = actor_net.ActorNet(state_dim, HIDDEN1_UNITS, action_dim)
critic = critic_net.CriticNet(state_dim, HIDDEN1_UNITS, action_dim)

# Initialize replay buffer

buff = ReplayBuffer(BUFFER_SIZE)      

def ddpg_player():
    env = Environment()
    r_t = 0.0
    sum_actions = 0.0
    steer = False
    for i in range(MAX_EPISODES):
        bot = Bot()
        s_t = np.zeros((state_dim,))
        s_t_1 = np.zeros((state_dim,))
        done = False
        counter = 0
        while bot.energy > 0:

            s_t[0:n] = bot.camera.depths
            s_t[n:n+3] = bot.hit, bot.energy, 1.0
            # Select action according to the curent policy and exploration noise    
            # add noise in the form of 1./(1.+i+j), decaying over episodes and
            # steps, otherwise a_t will be the same, since s is fixed per episode.
            
            index = counter % MINIBATCH_SIZE
            a_t = actor.predict_without_batch(s_t, index, target=False)
            a_t += 1./(1.+i+counter)
            a_t = np.clip(a_t, -action_bound, action_bound)

            # Execute action a_t and observe reward r_t and new state s_{t+1}
            energy, hit, distances, color = bot.forward(a_t[0], env, debug=False)

            r_t, counter, sum_actions = _get_reward(r_t, distances, hit, a_t[0][0], counter, sum_actions)
            s_t_1[0:n] = distances
            s_t_1[n:n+3] = hit, energy, 1.0

            if energy <= 0:
                done = True

            a = a_t[0][0]  # a_t is a 2d Array ergo we have to access it via 2 indices, maybe
            assert isinstance(a, np.float64), "Programming Error: Something went wrong while acceessing a_t"
            # Store transition in replay buffer
            buff.add(s_t, a, r_t, s_t_1, done)
            # If the no. of experiences (episodes) is larger than the mini batch size
            if buff.count() > MINIBATCH_SIZE:
                print(f"")
                # Sample a random batch 
                batch = buff.getBatch(MINIBATCH_SIZE)
                states_t = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                actions = np.expand_dims(actions, axis=1)  # make it (batch_size, action_dim)
                rewards = np.asarray([e[2] for e in batch])
                states_t_1 = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])
                # Setup y_is for updating critic
                y=np.zeros((len(batch), action_dim))

                a_tgt =actor.predict_with_batch(states_t_1, target=True)
                Q_tgt = critic.predict(states_t_1, a_tgt, target=True)
                
                for i in range(len(batch)):
                    if dones[i]:
                        y[i] = rewards[i]
                    else:
                        y[i] = rewards[i] + GAMMA*Q_tgt[i]    
                # Update critic by minimizing the loss
                critic.train(states_t, actions, rewards, y)
                # Update actor using sampled policy gradient
                a_for_dQ_da =actor.predict_with_batch(states_t, target=False)
                dQ_da = critic.evaluate_action_gradient(states_t,a_for_dQ_da)
                actor.train(states_t, dQ_da, ACTION_BOUND, index)
                
                # Update target networks
                actor.train_target(TAU)
                critic.train_target(TAU)
                
            counter += 1
            model = (actor.Win.T, actor.W.T, actor.Wout.T, 0, actor.leak, actor.relu, actor.relu)
            # Debug prints
            #print(f"Win: {actor.Win.T}")
            #print(f"W: {actor.W.T}")
            #print(f"W: {actor.Wout.T}")

            # return current model
            yield model

            s_t = s_t_1
            
            if done:
                print("Done!")
                break
        print("TOTAL REWARD @ " + str(counter) +"-th Episode:")
        print("")

def _get_reward(reward, distances, hit, action, counter, sum_actions):

    # reward function
    # split in two cases:
    # Hit - robot has it a wall
    # No Hit - robot is in free space
    # closeness to wall --> size of steering angle

    if hit is False:
        print("No hit!")
        sum_actions = 0.0
        counter = 0
        reward = 5.0
        if distances[31] > 0.4:
            if abs(action) > 0.5:
                reward = -5.0
            if abs(action) < 0.5:
                reward += 6.0
        if distances[31] < 0.4:
            if abs(action) > 1.0:
                reward += 1.0
            if abs(action) < 1.0:
                reward = -1.0
        if distances[31] < 0.2:
            if abs(action) > 2.0:
                reward = 10.0
            else:
                reward = -5.0

    else:
        print( "Hit!" )
        counter += 1
        if counter == 1:
            reward = -20.0
        else:
            reward -= 5.0
        sum_actions += np.sign(action)
        if abs(action) > 3.5:
            reward += (abs(sum_actions) * abs(action) * 0.01)

    print(f"Reward: {reward}")

    return reward, counter, sum_actions

                
    

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import numpy as np    
    from challenge import train, evaluate


    seed = 12345
    
    # Training (100 seconds)
    np.random.seed(seed)
    print(f"Starting training for 100 seconds (user time)")
    model = train(ddpg_player, timeout=100)

    # Evaluation
    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=True, seed=seed)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score: {score:.2f} ± {std:.2f}")        