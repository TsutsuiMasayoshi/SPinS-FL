# SPinS-FL
Source code of my bachelor thesis research, "SPinS-FL: Communication-Efficient Federated Subnetwork Learning." 

A communication-efficient **federated learning** algorithm utilizing **the Lottery Ticket Hypothesis**.

Accpeted as a conference paper in [IEEE Consumer Communications & Networking Conference (CCNC) 2023](https://ccnc2023.ieee-ccnc.org/).


# Federated Learning
<img src="https://user-images.githubusercontent.com/62000880/219865620-4dd3c489-b820-429c-a621-0f4883c81e47.png" width=400>


Federated learning (FL) is a distributed machine learning method in which edge devices collaboratively train a common model without disclosing their private training data. 

Unlike data centers, edge devices often stand in low-bandwidth traffic environments, which can be a bottleneck in the training process.

To solve this problem, we utilized the Edge-Popup Algorithm: a deep learning algorithm based on the Lottery Ticket Hypothesis.

# The Lottery Ticket Hypothesis
Pruning weights from neural networks have shown some success in the past; the required weight is only a part of the whole in a large neural network.

Based on this idea, Frankle and Carbin proposed the lottery ticket hypothesis, 
wherein a randomly weighted neural network contains a subnetwork whose test accuracy is comparable to that of the entire network
by updating the weights the same number of times.

# The Edge-Popup Algorithm
<img src="https://user-images.githubusercontent.com/62000880/219865817-3d41e2d1-e5c9-4dae-bb87-9e04d7ea7fa1.png" width=250>

Ramanujan et al.　further strengthened the argument in the lottery ticket hypothesis and proposed that a deep neural network has a hidden subnetwork that performs as well as the original network already trained.

Based on this strengthened hypothesis, Ramanujan et al. proposed the Edge-Popup algorithm, a novel method for **finding a well-performing subnetwork** by assigning **scores** to each weight in the network.

These scores indicate the priority of the respective weight to join a focused subnetwork.

By training these scores instead of the weights, a **supermask** that represents the location of the vital weights can be acquired.

# SPinS-FL
<img src="https://user-images.githubusercontent.com/62000880/219866088-b8da3110-15b8-4d64-afdd-ae8deae32263.png" width=400>

We first implement the Edge-Popup algorithm in FL naively. In the Edge-Popup algorithm, scores are trained instead of weights. Therefore, the score information should be communicated instead of weights.

In addition to this, we carefully selected the scores to communicate during the training and succeeded in reducing communication and calculation for the training.

The process is represented by the following images.

For the detail of this algorithm, please read the paper upcoming to be released.

<img src="https://user-images.githubusercontent.com/62000880/219866202-b0ef5260-c7ac-4487-b2d1-7d150459bd1a.png" width=250><img src="https://user-images.githubusercontent.com/62000880/219866191-effca74a-b27e-4b1d-a343-1dba2f3c67f8.png" width=300>


# Requirement
```
python==3.7.4
```
# Installation

```
pip install requirements.txt
```

# Usage
```
$ # at the top directory
$ mkdir data
$ ./runfl.sh
```
The accuracy is recorded in server.log for every round.

# Author

Masayoshi Tsutsui
The University of Tokyo 

# License

Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
