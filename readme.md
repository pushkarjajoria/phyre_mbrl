## Solving physical reasoning tasks using Model-Based Reinforcement learning

This is an ongoing project in which I try to build a Model-based RL agent which is able to solve physical reasoning problem.

I am using the [Phyre](https://phyre.ai/) benchmark from Facebook ai.

### Phyre 

For a in-depth understanding of the benchmark, please go through the official [research paper](https://arxiv.org/abs/1908.05656) from FB-AI but for a brief overview,

- Phyre contains a set of puzzles divided in to tiers -> templates -> puzzles.
- There are 2 tiers 
	1. Single ball puzzles: Puzzles that are guaranteed to be solved using a single ball
	2. Double ball puzzles: Puzzles that are guaranteed to be solved by using 2 balls
- We can sample puzzles from each tier.
- The action space is 3-Dimensional and 6-dimensional for tier 1 and tier 2 templates respectively. It is continuous.
- The output images from simulator are a sequence of snapshots in time.

#### Part-1 Predicting the model dynamics

![Sequence Prediction](/images/rnn_sequence_prediction.jpg)

#### Overall Algorithm

	Main Algorithm

	• Divide the available puzzles into train(1600) dev(400) test(500).
	• Get the cache for each of these puzzles.
		(Cache = 100k actions and their result on each of these puzzles)

	for each epoch:
		• create a randomly shuffled index list
		for each index in this randomly_shuffled_list:
			• increment the batch size
			• get the action status for this puzzle from cache
			• Get 9 random valid actions from this cache (Omitting invalid actions)
			• Simulate these actions on the environment to generate the target image sequence and append this to the dataset
			• Get one random solving action from the cache and append it to the dataset after simulating it on the environment.
			• To make all of the sequence of images of the same timestep, repeat the final image in the sequence
			• process_images to make them 100x100 grayscale
			
			if current_batch_size >= BATCH_SIZE:
				• get list of [X, Y, Y_reward, action]
				• Shuffle the dataset
				• train on this batch size
				• store the losses for use later

