# Predicting set-pieces (corner kicks) in soccer using Graph Neural Networks

## 1. Introduction
Set pieces play a pivotal role in soccer matches, often determining the outcome of games. According to the data collected by [Statsbomb](https://statsbomb.com/articles/soccer/changing-how-the-world-thinks-about-set-pieces/), set piece goals typically account for between 25% and 33% of all goals scored in the course of a season. It is important for teams and coaches to plan the tactics for set-pieces to maximize their chances of scoring a goal from these plays. For this project, we are specifically analyzing corner kicks, using Graph Neural Networks (GNNs) to predict the outcome of a corner kick based on the player positions and attributes. Introducing Corner Kick Studio - an interactive platform that allows users to visualize, analyze and enhance corner setups, providing a dynamic environment to simulate various tactical scenarios.

## 2. Dataset used
We have used 2 data sources for the project - [Statsbomb](https://github.com/statsbomb/open-data) and [EA FC 2024](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset). From statsbomb, we are using 2 apis - event data and 360 data. Event data captures information about different events that happened during the game (passes, goals, fouls etc.), and 360 data, which captures the frame of every event, including the coordinates (x,y) for every player in that frame. In addition to this, the data from EA FC 2024 provides details about player and team attributes like their attack and defense skill ratings which are used as features in our model.

![GNN Introduction](https://github.com/reojackson31/soccer-corner-kick-studio/assets/148725712/67e89838-5892-48cc-9ebb-2f28c490b6d9)

## 3. Graph Neural Network
Using Graph Neural Networks in PyTorch Geometric, the outcome of set pieces (corner kicks, for example) is predicted  with players as nodes and connections between players as edges for the network. The input dataset consists of frames that provide the location of players (from Statsbomb), player and team attributes (from EA FC 2024), and the outcome of the corner, whether it resulted in a shot or not. For the GNN model, we have used PyTorch geometric library to create a Graph Convolutional Network with 3 layers, and a linear classifier layer, with ReLU as the activation function, and cross entropy loss as the objective function to minimize.  The model is trained for 1000 epochs, and we reached an accuracy of 80% on the validation set.

![GNN Components](https://github.com/reojackson31/soccer-corner-kick-studio/assets/148725712/62009f46-8b8f-401a-9bac-2383ed96230f)

![GNN Modeling Process](https://github.com/reojackson31/soccer-corner-kick-studio/assets/148725712/3c2f0c82-2bf1-45bf-af1e-d86db5c8e5d4)


**References:**

1. Google DeepMind Blog, "TacticAI: AI Assistant for Football Tactics". Available at: [https://deepmind.google/discover/blog/tacticai-ai-assistant-for-football-tactics/](https://deepmind.google/discover/blog/tacticai-ai-assistant-for-football-tactics/)

2. Arxiv Preprint, "TacticAI: an AI assistant for football tactics". Available at: [https://arxiv.org/pdf/2310.10553](https://arxiv.org/pdf/2310.10553)

3. StatsBomb, "Changing How the World Thinks About Set Pieces". Available at: [https://statsbomb.com/articles/soccer/changing-how-the-world-thinks-about-set-pieces/](https://statsbomb.com/articles/soccer/changing-how-the-world-thinks-about-set-pieces/)


