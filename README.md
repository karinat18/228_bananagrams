# MCTS for Bananagrams

For our final project in Stanford's [Decision Making Under Uncertainty](https://aa228.stanford.edu/) course (AA228/CS238), we implemented Monte Carlo Tree Search to create an agent that plays a slightly modified version of Bananagrams. 

## Usage

Clone this repo and use the following command at root to run a game:
```julia
julia banana_mcts.jl
```
Note that the following Julia packages are required: \
Printf, Dates, Random, Parameters, Multisets, Graphs, GraphPlot, Plots, POMDPs, POMDPTools, QuickPOMDPs
\
\
While running a game, images of the board and bank are saved to a subfolder named with the date and time of game start. Subfolders of different game results are saved in a parent 'boards' folder.

## License

[MIT](https://choosealicense.com/licenses/mit/)
