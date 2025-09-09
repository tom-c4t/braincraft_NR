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
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
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
critic = critic_net.CriticNet(state_dim, action_dim, HIDDEN1_UNITS, action_dim)

# Initialize replay buffer

buff = ReplayBuffer(BUFFER_SIZE)      
step=0
reward_result=[]

def ddpg_player():
    env = Environment()
    for i in range(MAX_EPISODES):
        bot = Bot()
        s_t = np.zeros((state_dim,))
        s_t_1 = np.zeros((state_dim,))
        done = False
        total_reward = 0.
        counter = 0
        while bot.energy() > 0:
            loss=0

            s_t[0:n] = bot.camera.depths
            s_t[n:n+3] = bot.hit, bot.energy, 1.0
            # Select action according to the cuurent policy and exploration noise    
            # add noise in the form of 1./(1.+i+j), decaying over episodes and
            # steps, otherwise a_t will be the same, since s is fixed per episode.
            a_t = actor.predict(np.reshape(s_t,(1,3)), ACTION_BOUND, target=False)+1./(1.+i+counter)
            
            # Execute action a_t and observe reward r_t and new state s_{t+1}
            energy, hit, distances, color = bot.forward(a_t, env, debug=False)

            r_t = energy-s_t[-2]
            s_t_1[0:n] = distances
            s_t_1[n:n+3] = hit, energy, 1.0

            if energy <= 0:
                done = True
            
            # Store transition in replay buffer
            buff.add(s_t, a_t[0], r_t, s_t_1, done)
            
            # If the no. of experiences (episodes) is larger than the mini batch size
            if buff.count() > MINIBATCH_SIZE:
                # Sample a random batch 
                batch = buff.getBatch(MINIBATCH_SIZE)
                states_t = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                states_t_1 = np.asarray([e[3] for e in batch])
                # actions_t_1 = None # adapt replay buffer so that action in next state is returned as well
                dones = np.asarray([e[4] for e in batch])
                # Setup y_is for updating critic
                y=np.zeros((len(batch), action_dim))
                a_tgt=actor.predict(states_t_1, ACTION_BOUND, target=True)
                Q_tgt = critic.predict(states_t_1, a_tgt, target=True)
                
                for i in range(len(batch)):
                    if dones[i]:
                        y[i] = rewards[i]
                    else:
                        y[i] = rewards[i] + GAMMA*Q_tgt[i]    
                # Update critic by minimizing the loss
                loss += (critic.train(states_t, actions, rewards, states_t_1, y))/len(batch)
                # Update actor using sampled policy gradient
                a_for_dQ_da=actor.predict(states_t, ACTION_BOUND, target=False)
                dQ_da = critic.evaluate_action_gradient(states_t,a_for_dQ_da, loss)
                actor.train(states_t, dQ_da, ACTION_BOUND)
                
                # Update target networks
                actor.train_target(TAU)
                critic.train_target(TAU)
                
            counter += 1
                
            s_t = s_t_1
            total_reward += r_t    
            
            step += 1
            if done:
                "Done!"
                break
        reward_result.append(total_reward)
        print("TOTAL REWARD @ " + str(i) +"-th Episode:" + str(total_reward))
        print("Total Step: " + str(step))
        print("")
    # model = W_in, W, W_out, 0, leak, f, g
    model = (actor.Win, actor.W, actor.Wout, 0, actor.leak, f, g)
    return model

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
    score, std = evaluate(model, Bot, Environment, debug=False, seed=seed)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score: {score:.2f} ± {std:.2f}")        