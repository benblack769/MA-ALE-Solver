# MA-ALE-Solver

## Project goals

1. Implement an efficient basic shared-weights reinforcement learning agent in MA-ALE with pretrained image processing backbone
1. Set up metrics/losses/playthrough visualizations/etc
1. Experiment with exploration through clustering-diversity approaches

## Architecture

Basic components

1. Environment+model yields action/state pairs 
    * Debugging video generation
    * Reward statistics (average reward per game, average reward per step)
2. Q-Value approximation training loop with replay buffer
    * Large-ish, uniform replay buffer with uniform random eviction policy (keeps some older stuff with low probability)
    * Batched training loop
    * Q-value training loss statistics
    * Exports model to pt or environment/model loop to load
3. Diversity scoring
    * Isolation forest anomaly detection algorithm
    * Dataset = replay buffer with uniform random eviction policy -> embeddings (maybe used fixed embeddings from policy network)
    * Isolation forest outputs continuous outlier score -> lower the average position on the tree, the lower the score. 
    * Can reward lower-positioned elements over higher positioned ones on some sort of exponential function as an exploration target
