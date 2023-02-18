# SPinS-FL
Source code of my bachelor thesis research, "SPinS-FL: Communication-Efficient Federated Subnetwork Learning." 

A communication-efficient **federated learning** algorithm utilizing **the Lottery Ticket Hypothesis**.

Accpeted as a conference paper in [IEEE Consumer Communications & Networking Conference (CCNC) 2023](https://ccnc2023.ieee-ccnc.org/).


# Federated Learning
![iconicFL.pdf](https://github.com/MasayoshiTsutsui/SPinS-FL/files/10773702/iconicFL.pdf)

Federated learning (FL) is a distributed machine learning method in which edge devices collaboratively train a common model without disclosing their private training data. 

Unlike data centers, edge devices often stand in low-bandwidth traffic environments, which can be a bottleneck in the training process.

To solve this problem, we utilized the Edge-Popup Algorithm: a deep learning algorithm based on the Lottery Ticket Hypothesis.

# The Edge-Popup Algorithm
![EdgePopup.pdf](https://github.com/MasayoshiTsutsui/SPinS-FL/files/10773728/EdgePopup.pdf)

## the Lottery Ticket Hypothesis
Pruning weights from neural networks have shown some success in the past; the required weight is only a part of the whole in a large neural network.

Based on this idea, Frankle and Carbin proposed the lottery ticket hypothesis, 
wherein a randomly weighted neural network contains a subnetwork whose test accuracy is comparable to that of the entire network
by updating the weights the same number of times.

## Searching a Well-Performing Subnetwork
Ramanujan et al.　further strengthened the argument in the lottery ticket hypothesis and proposed that a deep neural network has a hidden subnetwork that performs as well as the original network already trained.

If this is true, then this hidden subnetwork has a significant advantage.

We can construct a network using only 2 things:

- a random seed for weight initialization and a supermask
- a bitmask indicating which weights belong to the subnetwork

Based on this strengthened hypothesis, Ramanujan et al. proposed the Edge-Popup algorithm, a novel method for **finding a well-performing subnetwork** by assigning **scores** to each weight in the network.

These scores indicate the priority of the respective weight to join a focused subnetwork.

By training these scores instead of the weights, a **supermask** that represents the location of the vital weights can be acquired.

# Features

TODO

# Requirement

TODO

# Installation

TODO

# Usage

TODO

# Note

TODO

# Author

Masayoshi Tsutsui
The University of Tokyo 

# License

TODO
