module Bananagrams

using Printf, Dates
using Parameters, Multisets
using Graphs, GraphPlot, Plots

export Tile, State, Action
export load_word_list, find_playable_word_list, play_on_board, init_state, is_terminal, see_board

##############################
# CUSTOM STRUCTS
##############################
mutable struct Tile
    letter::Char
    pos::Tuple{Int, Int}
    up::Int
    down::Int
    left::Int
    right::Int
end
Tile(letter::Char, pos::Tuple{Int, Int}) = Tile(letter, pos, 0, 0, 0, 0)

mutable struct State
    tiles::Vector{Tile}
    letter_bank::Vector{Char}
    occupied::Set{Tuple{Int, Int}}
    bunch::Vector{Char}
end
struct Action
    partial_word::String
    parent_index::Int
    direction::Symbol
end


##############################
# HELPER FUNCTIONS
##############################

# Actual bananagrams game tile counts
bananagrams_distribution = Dict(
    'A' => 13, 'B' => 3, 'C' => 3, 'D' => 6, 'E' => 18,
    'F' => 3, 'G' => 4, 'H' => 3, 'I' => 12, 'J' => 2,
    'K' => 2, 'L' => 5, 'M' => 3, 'N' => 8, 'O' => 11,
    'P' => 3, 'Q' => 2, 'R' => 9, 'S' => 6, 'T' => 9,
    'U' => 6, 'V' => 3, 'W' => 3, 'X' => 2, 'Y' => 3,
    'Z' => 2
)

function opposite_dir(dir_symbol::Symbol) ::Symbol
    """
    Get opposing directions
    """
    if dir_symbol == :up
        return :down
    elseif dir_symbol == :down
        return :up
    elseif dir_symbol == :left
        return :right
    else   # dir_symbol = right
        return :left
    end
end

function find_candidate_words(source_tile::Tile, letter_bank::Vector{Char}, dictionary::Set{String}) ::Vector{String}
    """
    Outputs a vector of (full) words that can be played given the source tile that already exists on the board
    Assume that the source tile can only be used as the first or last letter of the new word
    """
    all_letters = [source_tile.letter; letter_bank]
    set_letters = Multiset(all_letters)
    candidate_words = []

    for word in dictionary
        set_word = Multiset(collect(uppercase(word)))
        if ((uppercase(first(word)) == source_tile.letter) || uppercase(last(word)) == source_tile.letter) && issubset(set_word, set_letters)
            push!(candidate_words, word)
        end
    end
    return candidate_words
end

function free_direction(tile::Tile, dir_symbol::Symbol, candidate::String, occupied::Set{Tuple{Int, Int}})
    """
    Checks if the grid has enough free spots in a specific direction to fit a candidate word
    """
    if dir_symbol == :up
        dx = 0
        dy = 1
    elseif dir_symbol == :down
        dx = 0
        dy = -1
    elseif dir_symbol == :left
        dx = -1
        dy = 0
    else   # dir_symbol = right
        dx = 1
        dy = 0
    end

    for distance in 1:length(candidate)
        x = tile.pos[1] + distance * dx
        y = tile.pos[2] + distance * dy

        if dx == 0 # up or down. adjacent spots are left and right
            x1 = x + 1
            y1 = y
            x2 = x - 1
            y2 = y
        else # dy == 0. left or right. adjacent spots are up and down
            x1 = x
            y1 = y + 1
            x2 = x
            y2 = y - 1
        end

        if ((x, y) in occupied) || ((x1, y1) in occupied) || ((x2, y2) in occupied) # then collision
            return false
        elseif (distance == length(candidate)-1) && ((x+dx, y+dy) in occupied) # special case: current board perfectly wraps (makes a U shape) around candidate word. then another adjacent spot for last letter is in the direction dir_symbol (next to the first/last letter)
            return false
        end
    end
    return true
end

function check_no_collisions(tile::Tile, candidate::String, occupied::Set{Tuple{Int, Int}})
    """
    Verifies that there are no collisions and that the word is readable left to right or top to bottom
    """
    for (dir_symbol, dir_value) in (:up => tile.up, :down => tile.down, :left => tile.left, :right => tile.right)
        opposite = opposite_dir(dir_symbol)
        opposite_value = getfield(tile, opposite)
        if (dir_value == 0) && (opposite_value == 0) && (free_direction(tile, dir_symbol, candidate, occupied))
            # verify that the word will be read left to right, top to bottom
            if ((tile.letter == uppercase(last(candidate))) && ((dir_symbol == :up) || (dir_symbol == :left))) || ((tile.letter == uppercase(first(candidate))) && ((dir_symbol == :down) || (dir_symbol == :right)))
                return (true, dir_symbol)
            end
        end
    end
    return (false, nothing)
end

function construct_playable_word(candidate::String, tile::Tile, tiles::Vector{Tile}, direction::Symbol)
    parent = findfirst(t -> t == tile, tiles) # index of tile in vector of tiles

    # words need to be read left to right, top to bottom
    # these checks may fail if first and last letter of the word are the same ---------------------------------------------------------------!!!
    if ((direction == :up) && (tile.letter == uppercase(last(candidate)))) || ((direction == :left) && (tile.letter == uppercase(last(candidate))))
        partial_word = (reverse(candidate))[2:end] # partial word is candidate backwards, without the last letter of candidate
    elseif ((direction == :down) && (tile.letter == uppercase(first(candidate)))) || ((direction == :right) && (tile.letter == uppercase(first(candidate))))
        partial_word = candidate[2:end]  # partial word is candidate without first letter
    else
        return nothing
    end
    return Action(partial_word, parent, direction)
end

function random_bunch_arr(num_tiles::Int=35) ::Vector{Char}
    """ 
    Creates an array of random chars using the frequency defined in bananagrams_distribution
    Produces different result every time it's run
    """
    # Create a pool of letters
    pool = [repeat([k], v) for (k, v) in bananagrams_distribution]
    pool = collect(Iterators.flatten(pool))

    # Randomly select 35 letters from the pool
    chosen_letters = rand(pool, num_tiles)
    sort!(chosen_letters)
    return chosen_letters
end

function convert_bunch_type(bunch_arr::Vector{Char}; format::String="dict")
    """ 
    Make a random bunch represented as a dictionary with keys of all letters and their counts 
    """
    ASCII_zero = 64    # letter 'A' is 65

    # make a dictionary with all the letters and fill with the ones drawn
    bunch_dict = Dict{Char, Int}()
    for i in 1:26
        bunch_dict[Char(ASCII_zero+i)] = 0
    end
    for c in bunch_arr
        bunch_dict[c] += 1
    end

    # different output formats
    if format == "array"
        return bunch_arr
    elseif format == "string"
        return join(bunch_arr)
    elseif format == "dict"
        return bunch_dict
    else
        println("Format must be array, string, or dict")
        return
    end
end


##############################
# EXPORTED FUNCTIONS
##############################
function load_word_list(file_path) ::Set{String}
    """
    Input: file path to list of 3000 most common English words
    Output: words in the list that are 3-5 letters long
    """
    valid_dictionary = Set{String}()
    open(file_path, "r") do file
        for line in eachline(file)
            words = split(line)
            for word in words
                word = strip(word)
                if (3 <= length(word) <= 5) && islowercase(word[1])
                    push!(valid_dictionary, word)
                end
            end
        end
    end
    return valid_dictionary
end

function find_playable_word_list(tiles::Vector{Tile}, letter_bank::Vector{Char}, occupied::Set{Tuple{Int, Int}}, dictionary::Set{String}) ::Vector{Union{Action, Nothing}}
    """
    Outputs (partial) words that can be played along with their positions
    """
    playable_word_list = Vector{Union{Action, Nothing}}()
    for tile in tiles
        num_neighbors = count(n -> (n != 0), (tile.up, tile.down, tile.left, tile.right)) # if nonzero, then there is a neighbor in that direction
        if num_neighbors < 3
            candidate_words = find_candidate_words(tile, letter_bank, dictionary) # find possible candidate words that start/end with that letter tile
            for candidate in candidate_words
                candidate = uppercase(candidate)
                no_collisions, direction = check_no_collisions(tile, candidate, occupied)
                if no_collisions # then add that word the list of playable words
                    playable_word = construct_playable_word(candidate, tile, tiles, direction)
                    if isa(playable_word, Action)
                        push!(playable_word_list, playable_word)
                    end
                end
            end
        end
    end
    return playable_word_list
end

function play_on_board(partial_word::String, parent::Int, dir_symbol::Symbol, s::State)
    """
    Adds word to board, changing tiles, letter_bank, and occupied
    Assumes that the letters of partial_word exist in letter_bank
    """
    tiles = deepcopy(s.tiles)
    letter_bank = copy(s.letter_bank)
    occupied = deepcopy(s.occupied)
    bunch = copy(s.bunch)

    prev = parent
    for letter in partial_word
        prev_tile = tiles[prev]

        # decode relative position from prev tile
        if dir_symbol == :up
            dx = 0
            dy = 1
        elseif dir_symbol == :down
            dx = 0
            dy = -1
        elseif dir_symbol == :left
            dx = -1
            dy = 0
        else    # dir_symbol = right
            dx = 1
            dy = 0
        end

        # set absolute position
        x = prev_tile.pos[1] + dx
        y = prev_tile.pos[2] + dy

        # make a tile and link the previous tile to it
        curr_tile = Tile(letter, (x,y))
        setproperty!(curr_tile, opposite_dir(dir_symbol), prev)

        # add to tiles vector and mark space as occupied (play on board)
        push!(tiles, curr_tile)
        push!(occupied, (x, y))

        # remove one instance of the letter from the bank
        index_to_remove = findfirst(x -> x == curr_tile.letter, letter_bank)
        if index_to_remove !== nothing
            deleteat!(letter_bank, index_to_remove)
        end

        # link the current tile to the parent
        curr = length(tiles)
        setproperty!(prev_tile, dir_symbol, curr)

        prev = curr
    end
    return State(tiles, letter_bank, occupied, bunch)
end

function init_board(bunch::Vector{Char}, valid_dictionary, num_tiles=10)
    """ 
    Create a num_tiles-letter crossword to start the game
    Returns tiles, letter_bank, occupied
    """
    bunch_dict = convert_bunch_type(bunch, format="dict")

    # Initialize
    tiles = Vector{Tile}()
    occupied = Set{Tuple{Int, Int}}()
    letter_bank= Vector{Char}()

    # Temporarily set letter_bank to reflect the entire bunch in order to use the functions
    letter_bank = [k for (k, v) in bunch_dict if v > 0]

    # Choose a random letter from the bunch and play it at (0, 0)
    first_letter = rand(letter_bank)
    # println(first_letter)
    new_tile = Tile(first_letter, (0, 0))
    push!(tiles, new_tile)
    push!(occupied, (0, 0))

    # Remove one instance of the letter from the bank
    index_to_remove = findfirst(x -> x == first_letter, letter_bank)
    if index_to_remove !== nothing
        deleteat!(letter_bank, index_to_remove)
    end

    # Until you have played num_tiles, play words
    n = 1
    playable_word = Action("", 1, :right)
    while n < num_tiles
        # Find playable words
        playable_word_list = find_playable_word_list(tiles, letter_bank, occupied, valid_dictionary)
        if isempty(playable_word_list)
            # println("no words found!")
            return nothing, nothing, nothing, nothing
        end

        # Find a playable word that fits within num_tiles limit
        for (i, word) in enumerate(playable_word_list)
            playable_word = word
            if (n+length(playable_word.partial_word) <= num_tiles)
                break
            elseif (n+length(playable_word.partial_word) > num_tiles && i==length(playable_word_list))
                # println("words too long!")
                return nothing, nothing, nothing, nothing
            end
        end

        # Play the partial word on the board
        state = State(tiles, letter_bank, occupied, bunch)
        sp = play_on_board(playable_word.partial_word, playable_word.parent_index, playable_word.direction, state)
        tiles = sp.tiles
        letter_bank = sp.letter_bank
        occupied = sp.occupied
        n += length(playable_word.partial_word)
    end

    # Remove all played letters from the bunch
    for tile in tiles
        bunch_dict[tile.letter] -= 1
        deleteat!(bunch, findfirst(x->x==tile.letter, bunch))
    end
    
    # Reset the letter bank
    empty!(letter_bank)

    return tiles, letter_bank, occupied, bunch
end

function init_state(dictionary, BUNCH_TOT::Int=35, num_tiles=10)
    bunch = random_bunch_arr(BUNCH_TOT)
    bunch_dict = convert_bunch_type(bunch, format="dict")
    tiles = nothing
    bank = nothing
    occupied = nothing
    while true
        tiles, bank, occupied, bunch = init_board(bunch, dictionary, num_tiles)
        if !isnothing(tiles)
            break
        end
        bunch = random_bunch_arr(BUNCH_TOT)
        bunch_dict = convert_bunch_type(bunch, format="dict")
    end
    return State(tiles, bank, occupied, bunch)
end

function is_terminal(s::State, dictionary, BANK_MAX)
    if length(find_playable_word_list(s.tiles, s.letter_bank, s.occupied, dictionary)) == 0
        if length(s.letter_bank) == BANK_MAX || length(s.bunch) == 0
            return true
        end
    end
    return false
end

function see_board(tiles::Vector{Tile}, letter_bank::Vector{Char}, bunch_size::Int, folder_path; save=false)
    """
    Visualize (and save .png of) the board and letter bank
    """
    # Flag for an non-empty bank
    has_bank = false
    if length(letter_bank) > 0
        has_bank = true
    end

    # Extract positions from Tile struct for plotting
    board_pos = [(tile.pos[1], tile.pos[2]) for tile in tiles]
    board_xs, board_ys = collect(first.(board_pos)), collect(last.(board_pos))

    max_x=0
    max_y=0
    if has_bank
        # Plot letter_bank above board
        bank_pos = [(minimum(board_xs)+i-1, maximum(board_ys)+1.5) for i in 1:length(letter_bank)]
        bank_xs, bank_ys = collect(first.(bank_pos)), collect(last.(bank_pos))
        # Value limits
        max_x = max(maximum(board_xs), maximum(bank_xs))
        max_y = maximum(bank_ys)
    else
        max_x = maximum(board_xs)
        max_y = maximum(board_ys)
    end

    # Scale markersize to fill 1 axis unit
    buffer = 0.5
    markersize_in_units = 0.5
    x_range = (max_x-minimum(board_xs))
    y_range = (max_y-minimum(board_ys))
    x_pixels_per_unit = 525 / (x_range + 2*buffer)  # default plot size is (600, 400)
    y_pixels_per_unit = 325 / (y_range + 2*buffer)
    markersize_pixels = markersize_in_units * min(x_pixels_per_unit, y_pixels_per_unit)

    # Plot (can add axis=([], false) to remove axis labels and grid)
    scatter(board_xs, board_ys, ratio=1, label="", axis=([], false),
    shape=:square, markersize=markersize_pixels, color=:black)
    if has_bank
        scatter!(bank_xs, bank_ys, ratio=1, label="",
        shape=:square, markersize=markersize_pixels, color=:royalblue, markerstrokecolor=:royalblue)
        plot!([minimum(board_xs)-buffer, max_x+buffer], [max_y-0.75, max_y-0.75], ls=:solid, lw=2, lc=:royalblue, legend=false)
    end
        plot!()

    # Add tile labels
    for tile in tiles
        annotate!(tile.pos[1], tile.pos[2], text(tile.letter, 14, :white, :center))
    end
    if has_bank
        for i in 1:length(letter_bank)
            annotate!(bank_pos[i][1], bank_pos[i][2], text(letter_bank[i], 14, :white, :center))
        end
    end

    # Set axis limits with buffer to not cut off marker
    xlims!(minimum(board_xs) - buffer, max_x + buffer)
    ylims!(minimum(board_ys) - buffer, max_y + buffer)

    # Customize plot appearance (optional)
    title!("Bananagrams Bank and Board, Unseen Tiles: $bunch_size")

    # Display the plot
    display(Plots.plot!())  # Note: Plots.plot!() returns the current plot

    # Save the plot to a PNG file with a timestamp
    if save
        timestamp = Dates.format(now(), "mmdd_HHMM_SS_sss")
        if !isdir(folder_path)
            mkpath(folder_path)
        end
        savefig(joinpath(folder_path, "board_$timestamp.png"))
    end
end

end