# Proximal Policy Optimization(PPO)
## Overview
Here is the original paper of PPO: https://arxiv.org/pdf/1707.06347.pdf

The implementation here is somehow based on the Unity ML Agent's python implementation. https://github.com/Unity-Technologies/ml-agents

## Algorithm 1 PPO, Actor-Critic Style
### input and output
The input of PPO neural network should be the states.

The output of a PPO actor neural network will be a distribution of policy's actions. For example, if the action is discrete, the output can be the probabilities of each action; if the action is continuous, the output can be the mean and variance of a normal distribution.

The action to take will be sampled based on the output distribution.

The output of a PPO critic neural network will be the value of each state.
 
### training data
The training data is from the PPO agent playing the game. Those data needs to be collected at each step
* state
* action taken sample from the output distribution of output from actor network
* the probability of the action taken based on the output distribution of output from actor network
* output value from critic network
* reward obtained after the action is taken
After end of each game, some other data will be calculated based on the recorded data:
* advantage: ![advantage equation](https://github.com/tcmxx/CNTKUnityTools/blob/master/Docs/Images/advantage-equation.png)
* target value = advantage + recorded value
### Loss Function:
L = L_policy + L_value + L_entropy
* L_value = (target value - value from NN)^2
* entropy: For normal distribution: ![entropy for normal distribution](https://wikimedia.org/api/rest_v1/media/math/render/svg/5c47c048d3fbf311a0b8af942f44f02908bec393)
* L_policy: See equation 7 and 6 on: https://arxiv.org/pdf/1707.06347.pdf . Note that the old probability is the recorded probability. The new probability if the current probability of the recorded action(this is a little different from Unity's implementation).
### pseudo code
```python
for iteration=1, 2, . . . do:
  for actor=1, 2, . . . , N do:
    Run policy in environment for T timesteps or until the end of game
    Compute advantage estimates A_1, . . . , A_T
  Optimize loss: L, with K epochs and minibatch size M ≤ NT and update the policy.
```

## Usage:
1. Define your environment and implement the interface IRLEnvironment.
2. Create your neural network for PPO.

```csharp
var network = new PPONetworkContinuousSimple(stateSize, actionSize, numOflayers, hiddenSize, DeviceDescriptor.GPUDevice(0),0.01f);
```

 - PPONetworkContinuousSimple is a simple dense neural network implementation. You can also implement your own network by inherit the abstract class ```public abstract class PPONetwork```.

3. Create a `PPOModel` using the neural network. `PPOModel` will add variables that are needed for training on top of your neural network. It also provides some helper functions for evaluation and so on.
```csharp
var model = new PPOModel(network);
```

4. Create the PPO trainer. The PPO `TrainerPPOSimple` is a helper class to train the PPO network. It helps step the environment and record the reward,states etc, and calculate the discounted advantage after each episode. 
```csharp
trainer = new TrainerPPOSimple(model, LearnerDefs.AdamLearner(learningRate), bufferSize, maxStepHorizon);
```
- The `buffersize` is the capacity of the buffer that records the data used in the trainer. It should be at least maxPossibleStepOfEachGame*numOfGamesToRunBeforeEachTrain. 
- The `maxStepHorizon` is the max steps of each game before calculate the discounted advantage. One game is to be end when this is reached.

5. Run the steps every loop until the game agent is winning the game. 
```csharp
        trainer.Step(environment);//step the environment using the trainer.
        bool reset = trainer.Record(environment);//record the information after the step, and return whether should reset the environment.
        //reset if yes
        if (reset)
        {
            environment.Reset();
            // If certain number of episodes' data is collected, start the train
            // Ideally only one episode of run is needed and instead multiply agent will run in parallel to get more data.
            if (episodesThisTrain >= episodeToRunForEachTrain)
            {
                //train with all collected data.
                //the trainer will randomize the data's order and train using all of them with a minibatch size
                //and this process will be done for iterationForEachTrain times.
                trainer.TrainAllData(minibatch, iterationForEachTrain);
                print("Training Loss:" + trainer.LastLoss); //print the loss
                //clear the data
                trainer.ClearData();
                episodesThisTrain = 0;
            }

        }
```
