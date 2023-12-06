using Printf, Random
using POMDPs, POMDPTools
using QuickPOMDPs: QuickMDP

include("Bananagrams.jl")
using .Bananagrams


# Global variables
dict_file = "3000_common_words.txt"
dictionary = load_word_list(dict_file)
BANK_MAX = 8
BUNCH_TOT = 40
bunch = random_bunch_arr(BUNCH_TOT)
bunch_dict = convert_bunch_type(bunch, format="dict")

# Reward values
turn_penalty = -1
leftover_penalty = -10   # per tile
none_left_reward = 100


# Define bananagrams MDP
bananagrams = QuickMDP(
    statetype = State,
    actiontype = Union{Action, Nothing},
    # obstype = State,   # no obs in MDP, just to prevent getting a warning
    discount = 0.95,

    isterminal = function (s)
        if length(find_playable_word_list(s.tiles, s.letter_bank, s.occupied, dictionary)) == 0
            if length(s.letter_bank) == BANK_MAX || length(bunch) == 0
                return True
            end
        end
        return false
    end,

    actions = function (s)
        actions = find_playable_word_list(s.tiles, s.letter_bank, s.occupied, dictionary)
        if length(s.letter_bank) < BANK_MAX && length(bunch) > 0
            push!(actions, nothing)  # nothing = draw a tile
        end
        return actions
    end,

    transition = function (s, a)
        if a == nothing   # draw tile from bunch and add to bank
            tile = rand(bunch)
            deleteat!(bunch, findfirst(x->x==tile, bunch))
            push!(s.letter_bank, tile)
        else
            play_on_board(a.partial_word, a.parent_index, a.direction, s.tiles, s.letter_bank, s.occupied)
        end
        return s
    end,

    reward = function (s, a, sp)
        r = turn_penalty
        if isterminal(sp)
            num_leftover = length(bank) + length(bunch)
            if num_leftover == 0
                r += none_left_reward
            else
                r += num_leftover*leftover_penalty
            end
        end
        return r
    end,

    initialstate = Deterministic(init_state(bunch_dict, dictionary, BUNCH_TOT)),
)


# Define MCTS struct and functions

struct MonteCarloTreeSearch
    𝒫   # problem
    N   # dictionary of visit counts
    Q   # dictionary of action value estimates
    d   # depth
    m   # number of simulations
    c   # exploration constant
    U   # value function estimate
end

bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)
function explore(π::MonteCarloTreeSearch, s)
    𝒜, N, Q, c = actions(π.𝒫, s), π.N, π.Q, π.c
    Ns = sum(N[(s,a)] for a in 𝒜)
    return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), 𝒜)
end

function simulate!(π::MonteCarloTreeSearch, s, d=π.d)
    if d <= 0
        return π.U(s)
    end
    𝒫, N, Q, c = π.𝒫, π.N, π.Q, π.c
    𝒜, γ = actions(𝒫, s), discount(𝒫)
    if !haskey(N, (s, first(𝒜)))
        for a in 𝒜
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return π.U(s)
    end
    a = explore(π, s)
    sp = transition(𝒫, s, a)
    r = reward(𝒫, s, a, sp)
    q = r + γ*simulate!(π, sp, d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    return q
end

function (π::MonteCarloTreeSearch)(s)
    for k in 1:π.m
        simulate!(π, s)
    end
    return argmax(a->π.Q[(s,a)], actions(π.𝒫, s))
end

# Value estimate from random rollout
function U(s)
    sim = RolloutSimulator(rng=Random.default_rng())
    policy = RandomPolicy(bananagrams)             # generates actions at each state and randomly chooses one
    return simulate(sim, bananagrams, policy, s)   # returns reward from rollout from state s; ends when isterminal is true
end

N = Dict{Tuple{State, Union{Action, Nothing}}, Int}()
Q = Dict{Tuple{State, Union{Action, Nothing}}, Float64}()
d = 10
m = 100
c = 100    # d, m, c values used in textbook example

π = MonteCarloTreeSearch(bananagrams, N, Q, d, m, c, U)

init_tiles = Vector{Tile}()
init_bank = Vector{Char}()
init_occupied = Set{Tuple{Int, Int}}()
while true
    init_tiles, init_bank, init_occupied = init_board(bunch_dict, dictionary)
    if !isnothing(init_tiles)
        break
    end
end
s = State(init_tiles, init_bank, init_occupied)
while !isterminal(π.𝒫, s)
    a = π(s)   # action to take accord to MCTS
    transition(π.𝒫, s, a)
end

see_board(s.tiles, s.letter_bank, save=true)


# # POMDPs API: https://juliapomdp.github.io/POMDPs.jl/latest/api/#API-Documentation
# println("γ: ", discount(bananagrams))   # usage test
# println("Bunch: ", bunch)

# # Test board visualization: 
# # NOTE: saving .png file works but board does not pop up when display() is called
# my_tiles = Vector{Tile}()
# S_tile = Tile('S', (1,3), 0, 4, 0, 2)
# I_tile = Tile('I', (2,3), 0, 0, 1, 3)
# T_tile = Tile('T', (3,3), 0, 0, 2, 0)
# P_tile = Tile('P', (1,2), 1, 5, 0, 0)
# A_tile = Tile('A', (1,1), 4, 0, 0, 0)
# push!(my_tiles, S_tile)
# push!(my_tiles, I_tile)
# push!(my_tiles, T_tile)
# push!(my_tiles, P_tile)
# push!(my_tiles, A_tile)
# occupied = Set{Tuple{Int, Int}}([(1,1), (1,2), (1,3), (2,3), (3,3)])
# my_bank = ['C', 'R', 'T', 'S', 'A', 'G']
# test_state = State(my_tiles, occupied, my_bank)
# see_board(my_tiles, my_bank, save=true)
