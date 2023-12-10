using Printf, Dates, Random
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
        return is_terminal(s, dictionary, BANK_MAX)
    end,

    actions = function (s)
        actions = find_playable_word_list(s.tiles, s.letter_bank, s.occupied, dictionary)
        if length(s.letter_bank) < BANK_MAX && length(s.bunch) > 0
            push!(actions, nothing)  # nothing = draw a tile
        end
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
    Ns = sum(N[(s,a)] for a in 𝒜)
    return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), 𝒜)
end

function simulate!(π::MonteCarloTreeSearch, s, d=π.d)
    if d <= 0
        return π.U(π.𝒫, s)
    end
    𝒫, N, Q, c = π.𝒫, π.N, π.Q, π.c
    𝒜, γ = actions(𝒫, s), discount(𝒫)

    if isterminal(𝒫, s)
        return π.U(π.𝒫, s)
    end
    if !haskey(N, (s, first(𝒜)))
        for a in 𝒜
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return π.U(π.𝒫, s)
    end

    a = explore(π, s)
    sp = rand(transition(𝒫, s, a))
    r = reward(𝒫, s, a, sp)
    q = r + γ*simulate!(π, sp, d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    return q
end

function (π::MonteCarloTreeSearch)(s)
    for k in 1:π.m
        print(k, " ")
        simulate!(π, s)
    end
    println("")
    return argmax(a->π.Q[(s,a)], actions(π.𝒫, s))
end

# Value estimate from random rollout
function rand_rollout(𝒫::QuickMDP, s)
    if isterminal(𝒫, s)
        return 0
    end
    𝒜, γ = actions(𝒫, s), discount(𝒫)
    num_actions = length(𝒜)
    a = 𝒜[rand(1:num_actions)]
    sp = rand(transition(𝒫, s, a))
    r = reward(𝒫, s, a, sp)
    q = r + γ*rand_rollout(𝒫, sp)
    return q
end


# Play the game
function main()
    N = Dict{Tuple{State, Union{Action, Nothing}}, Int}()
    Q = Dict{Tuple{State, Union{Action, Nothing}}, Float64}()
    d = 5
    m = 5
    c = 100

    π = MonteCarloTreeSearch(bananagrams, N, Q, d, m, c, rand_rollout)

    s = rand(initialstate(π.𝒫))
    folder_path = "boards/" * Dates.format(now(), "mmdd_HHMM")
    see_board(s.tiles, s.letter_bank, length(s.bunch), folder_path, save=true)

    turn_count = 1
    while !isterminal(π.𝒫, s)
        println("Turn number: ", turn_count)
        println("Simulation progress: ")
        a = π(s)   # action to take accord to MCTS
        s = rand(transition(π.𝒫, s, a))
        see_board(s.tiles, s.letter_bank, length(s.bunch), folder_path, save=true)
        turn_count += 1
    end
end
main()