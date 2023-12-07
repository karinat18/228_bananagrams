using Printf, Random.Random
# Random.seed!(1234);
# rng = Random.default_rng(0)
# rng = Random.GLOBAL_RNG
# rng = MersenneTwister(0)
# rng = RandomDevice()
# rng = AbstractRNG
# println(typeof(rng))
using POMDPs, POMDPTools
using QuickPOMDPs: QuickMDP

include("Bananagrams.jl")
using .Bananagrams

# Global variables
dict_file = "3000_common_words.txt"
dictionary = load_word_list(dict_file)
BANK_MAX = 8
BUNCH_TOT = 40

# Reward values
turn_penalty = -1
leftover_penalty = -10   # per tile
none_left_reward = 100


# Define bananagrams MDP
bananagrams = QuickMDP(
    statetype = State,
    actiontype = Union{Action, Nothing},
    discount = 0.95,

    isterminal = function (s)
        # println("Num playable words: ", length(find_playable_word_list(s.tiles, s.letter_bank, s.occupied, dictionary)))
        if length(find_playable_word_list(s.tiles, s.letter_bank, s.occupied, dictionary)) == 0
            if ((length(s.letter_bank) == BANK_MAX) || (length(s.bunch) == 0))
                # println("IS TERMINAL")
                return true
            end
        end
        # println("NOT TERMINAL")
        return false
    end,

    actions = function (s)
        actions = find_playable_word_list(s.tiles, s.letter_bank, s.occupied, dictionary)
        if length(s.letter_bank) < BANK_MAX && length(s.bunch) > 0
            push!(actions, nothing)  # nothing = draw a tile
        end
        # println("Actions to return:", actions)
        return actions
    end,

    transition = function (s, a)
        if a === nothing   # draw tile from bunch and add to bank
            sp = State(deepcopy(s.tiles), copy(s.letter_bank), deepcopy(s.occupied), copy(s.bunch))
            new_tile = rand(s.bunch)
            deleteat!(sp.bunch, findfirst(x->x==new_tile, sp.bunch))
            push!(sp.letter_bank, new_tile)
        else
            sp = play_on_board(a.partial_word, a.parent_index, a.direction, s)
        end
        return Deterministic(sp)
    end,

    reward = function (s, a, sp)
        r = turn_penalty
        if is_terminal(sp, dictionary, BANK_MAX)
            num_leftover = length(sp.letter_bank) + length(sp.bunch)
            if num_leftover == 0
                r += none_left_reward
            else
                r += num_leftover*leftover_penalty
            end
        end
        return r
    end,

    initialstate = Deterministic(init_state(dictionary, BUNCH_TOT)),
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
    # println("EXPLORE recieved actions: ", 𝒜)
    Ns = sum(N[(s,a)] for a in 𝒜)
    return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), 𝒜)
end

function simulate!(π::MonteCarloTreeSearch, s, d=π.d)
    if d <= 0
        return π.U(π.𝒫, s)
    end
    𝒫, N, Q, c = π.𝒫, π.N, π.Q, π.c
    𝒜, γ = actions(𝒫, s), discount(𝒫)
    # println("SIM recieved actions: ", 𝒜)
    if isterminal(𝒫, s)
        # println("SIM is terminal")
        return π.U(π.𝒫, s)
    end
    if !haskey(N, (s, first(𝒜)))
        # println("Unvisited state: ", s)
        for a in 𝒜
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return π.U(π.𝒫, s)
    end
    # println("Visited state: ", s)
    a = explore(π, s)
    sp = rand(transition(𝒫, s, a))
    r = reward(𝒫, s, a, sp)
    q = r + γ*simulate!(π, sp, d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    # println("Q dict: ", Q)
    return q
end

function (π::MonteCarloTreeSearch)(s)
    for k in 1:π.m
        println("Simulation ", k)
        # println("num tiles: ", length(s.tiles))
        # println("num bank: ", length(s.letter_bank))
        # println("num bunch: ", length(s.bunch))
        simulate!(π, s)
    end
    return argmax(a->π.Q[(s,a)], actions(π.𝒫, s))
end

# Value estimate from random rollout
function rand_rollout(𝒫::QuickMDP, s)
    if isterminal(𝒫, s)
        # println("ROLLOUT is terminal")
        return 0
    end
    𝒜, γ = actions(𝒫, s), discount(𝒫)
    # println("ROLLOUT recieved actions: ", 𝒜)
    num_actions = length(𝒜)
    a = 𝒜[rand(1:num_actions)]
    # println("ROLLOUT chosen action: ", a)
    sp = rand(transition(𝒫, s, a))
    r = reward(𝒫, s, a, sp)
    q = r + γ*rand_rollout(𝒫, sp)
    # println("q from rollout: ", q)
    return q
end

function main()
    N = Dict{Tuple{State, Union{Action, Nothing}}, Int}()
    Q = Dict{Tuple{State, Union{Action, Nothing}}, Float64}()
    d = 3
    m = 5
    c = 100    # d, m, c values used in textbook example

    π = MonteCarloTreeSearch(bananagrams, N, Q, d, m, c, rand_rollout)

    s = rand(initialstate(π.𝒫))
    see_board(s.tiles, s.letter_bank, save=true)
    println("initial state: ", s)

    while !isterminal(π.𝒫, s)
        # println("MAIN state: ", s)
        a = π(s)   # action to take accord to MCTS
        s = rand(transition(π.𝒫, s, a))
        # println("MAIN next state: ", s)
        # s = sp
        see_board(s.tiles, s.letter_bank, save=true)
    end
    println("final state: ", s)

end

main()


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
