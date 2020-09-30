## Solving physical reasoning tasks using Model-Based Reinforcement learning

This is an ongoing project in which I try to build a Model-based RL agent which is able to solve physical reasoning problem.

I am using the [Phyre](https://phyre.ai/) benchmark from Facebook ai.

#### Phyre 

For a in-depth understanding of the benchmark, please go through the official [research paper](https://arxiv.org/abs/1908.05656) from FB-AI but for a brief overview,

- Phyre contains a set of puzzles divided in to tiers -> templates -> puzzles.
- There are 2 tiers 
	1. Single ball puzzles: Puzzles that are guaranteed to be solved using a single ball
	2. Double ball puzzles: Puzzles that are guaranteed to be solved by using 2 balls
- We can sample puzzles from each tier.
- The action space is 3-Dimensional and 6-dimensional for tier 1 and tier 2 templates respectively. It is continuous.
- The output images from simulator are a sequence of snapshots in time.

#### Part-1 Predicting the model dynamics

[Sequence Prediction](/images/rnn_sequence_prediction.jpg)
