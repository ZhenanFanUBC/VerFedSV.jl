########################################################################
# Synchronous Valuator
########################################################################

mutable struct Valuator{T1<:Int64, T2<:Vector{SparseMatrixCSC{Float64, Int64}}, T3<:Vector{Int64}, T4<:Bool}  
    Ytrain::T3
    num_classes::T1
    num_clients::T1
    num_epoches::T1
    batch_size::T1
    embedding_matrices::T2
    is_approx::T4
    time_points::T3
    sample_points::T3
    function Valuator(Ytrain::Vector{Int64}, config::Dict{String, Union{Int64, Float64}}; is_approx::Bool = false)
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        num_epoches = config["num_epoches"]
        batch_size = config["batch_size"]
        embedding_matrices = Vector{SparseMatrixCSC{Float64, Int64}}()
        N = length(Ytrain); num_batches = div(N, batch_size); 
        T = num_epoches * num_batches 
        if is_approx
            time_points = collect(1:200)
            sample_points = randperm(N)[1:5000]
        else
            time_points = collect(1:T)
            sample_points = collect(1:N)
        end
        for _ = 1:num_clients
            push!(embedding_matrices, spzeros(N, T*num_classes))
        end
        new{Int64, Vector{SparseMatrixCSC{Float64, Int64}}, Vector{Int64}, Bool}(Ytrain, num_classes, num_clients, num_epoches, batch_size, embedding_matrices, is_approx, time_points, sample_points)
    end
end

# server send embeddings to valuator
function send_embedding(s::Server, v::Valuator, round::Int64)
    num_clients = s.num_clients
    # num_classes = s.num_classes
    Threads.@threads for i = 1:num_clients
        # v.embedding_matrices[i][ (round-1)*num_classes+1 : round*num_classes, s.batch] = s.embeddings[i,:,:]
        update_embedding_matrix!(v.embedding_matrices[i], s.batch, s.embeddings[i,:,:], round)
    end
end

# complete the embedding matrices
function complete_embedding_matrices(v::Valuator, r::Int64)
    num_clients = v.num_clients
    Factors = Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}(undef, num_clients)
    Threads.@threads for i=1:num_clients
        @printf "start completing embedding matrix for client %i \n" i
        rows = vcat( [collect((t-1)*v.num_classes+1: t*v.num_classes) for t in v.time_points]...)
        columns = v.sample_points
        H = copy(v.embedding_matrices[i]')[rows, columns]
        X, Y = complete_matrix(H, r)
        Factors[i] = (X,Y)
    end
    return Factors
end

# roundly utility function
function Uₜ(v::Valuator, S::Vector{Int64}, t::Int64, Factors::Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}, b::Vector{Float64})
    sample_points = v.sample_points
    sample_size = length(sample_points)
    num_classes = v.num_classes
    Sc = setdiff( collect(1:v.num_clients), S )
    last = ((t-2)*num_classes+1) : (t-1)*num_classes
    current = ((t-1)*num_classes+1) : t*num_classes
    sum_embeddings_last = zeros(num_classes, sample_size)
    sum_embeddings_current = zeros(num_classes, sample_size)
    for i in Sc
        (X, Y) = Factors[i]
        sum_embeddings_last .+= X[last, :] * Y
        sum_embeddings_current .+= X[last, :] * Y
    end
    for i in S
        (X, Y) = Factors[i]
        sum_embeddings_last .+= X[last, :] * Y
        sum_embeddings_current .+= X[current, :] * Y
    end
    train_loss_last = 0.0
    train_loss_current = 0.0
    for i = 1:sample_size
        y = v.Ytrain[sample_points[i]]
        emb_last = sum_embeddings_last[:, i] + b
        emb_current = sum_embeddings_current[:, i] + b
        pred_last = softmax(emb_last)
        pred_current = softmax(emb_current)
        train_loss_last += neg_log_loss(pred_last, y)
        train_loss_current += neg_log_loss(pred_current, y)
    end
    return (train_loss_last - train_loss_current) / sample_size
end

# utility function
function utility(v::Valuator, S::Vector{Int64}, Factors::Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}, b::Vector{Float64})
    T = length(v.time_points)
    u = 0.0
    for t = 2:T
        u += Uₜ(v, S, t, Factors, b)
    end
    return u
end

# compute shapley value
function compute_shapley_value(v::Valuator, Factors::Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}, b::Vector{Float64})
    num_clients = v.num_clients 
    @printf "start computing all utilities \n"
    all_utilities = Dict{Vector{Int64},Float64}()
    all_subsets = collect( powerset(collect(1:num_clients), 1) )
    Threads.@threads for S in all_subsets
        all_utilities[S] = utility(v, S, Factors, b)
    end
    shapley_values = Vector{Float64}()
    for i in 1:num_clients
        @printf "start computing shapley value for client %i \n" i
        # power set of [M] \ {i}
        all_subsets_i = collect( powerset(setdiff(collect(1:num_clients), [i]), 1) )
        si = 0.0
        for S in all_subsets_i
            c = 1 / binomial(num_clients - 1, length(S))
            # U(S ∪ i)
            u1 = all_utilities[ sort(vcat(S, [i])) ]
            # U(S)
            u2 = all_utilities[ S ]
            si += c*(u1 - u2)
        end
        push!(shapley_values, si)
    end
    return shapley_values
end

# compute shapley value with monte carlo 
function compute_shapley_value_monte_carlo(v::Valuator, Factors::Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}, b::Vector{Float64})
    @printf "start computing Shapley value with Monte Carlo \n"
    num_clients = v.num_clients 
    K = 10*num_clients
    perm = collect(1:num_clients)
    shapley_values = zeros(num_clients)
    Threads.@threads for k = 1:K
        @printf "permutation %d \n" k
        # generate random permutation
        randperm!(perm)
        # previous utility
        pre = 0.0
        # add utilities to shapley values
        for i = 1:num_clients
            idx = perm[i]
            curr = utility(v, perm[1:i], Factors, b)
            shapley_values[idx] += (curr - pre) / K
            pre = curr
        end
    end
    return shapley_values
end

# compute singular values of the embedding matrices
function compute_svds_embedding_matrices(v::Valuator, r::Int64)
    num_clients = v.num_clients
    singular_values = Vector{Vector{Float64}}(undef, num_clients)
    for i=1:num_clients
        @printf "start computing singular values for client %i \n" i
        ss = svds(v.embedding_matrices[i], nsv=r)[1].S
        singular_values[i] = ss
    end
    return singular_values
end

########################################################################
# Asynchronous Valuator
########################################################################
mutable struct AsynValuator{T1<:Int64, T2<:Float64, T3<:Vector{T1}, T4<:Array{T2, 4}, T5<:Vector{T3}}  
    Ytrain::T3
    num_classes::T1
    num_clients::T1
    embeddings_all_rounds::T4
    all_subsets::T5
    Δt::T2
    current_round::T1
    function AsynValuator(Ytrain::Vector{Int64}, Δt::Float64, config::Dict{String, Union{Int64, Float64}})
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        time_limit = config["time_limit"]
        num_rounds = Int64(time_limit / Δt)
        embeddings_all_rounds = zeros(Float64, num_rounds, num_clients, num_classes, length(Ytrain))
        all_subsets = collect( powerset(collect(1:num_clients), 1) )
        current_round = 1
        new{Int64, Float64, Vector{Int64}, Array{Float64, 4}, Vector{Vector{Int64}}}(Ytrain, num_classes, num_clients, embeddings_all_rounds, all_subsets, Δt, current_round)
    end
end

# server send embeddings to valuator
function send_embedding(s::AsynServer, v::AsynValuator)
    v.embeddings_all_rounds[v.current_round, :, :, :] .= s.embeddings
    v.current_round += 1
end

# roundly utility function
function Uₜ(v::AsynValuator, S::Vector{Int64}, t::Int64, b::Vector{Float64})
    train_size = length(v.Ytrain)
    num_classes = v.num_classes
    Sc = setdiff( collect(1:v.num_clients), S )
    sum_embeddings_last = reshape( sum( v.embeddings_all_rounds[t-1, :, :, :], dims=1), num_classes, train_size)
    sum_embeddings_current = reshape( sum( v.embeddings_all_rounds[t-1, Sc, :, :], dims=1), num_classes, train_size) + reshape( sum( v.embeddings_all_rounds[t, S, :, :], dims=1), num_classes, train_size)
    train_loss_last = 0.0
    train_loss_current = 0.0
    for i = 1:train_size
        y = v.Ytrain[i]
        emb_last = sum_embeddings_last[:, i] + b
        emb_current = sum_embeddings_current[:, i] + b
        pred_last = softmax(emb_last)
        pred_current = softmax(emb_current)
        train_loss_last += neg_log_loss(pred_last, y)
        train_loss_current += neg_log_loss(pred_current, y)
    end
    return (train_loss_last - train_loss_current) / train_size
end

# utility function
function utility(v::AsynValuator, S::Vector{Int64}, b::Vector{Float64})
    T = size(v.embeddings_all_rounds, 1)
    u = 0.0
    for t = 2:T
        u += Uₜ(v, S, t, b)
    end
    return u
end

# compute shapley value
function compute_shapley_value(v::AsynValuator, b::Vector{Float64})
    num_clients = v.num_clients 
    @printf "start computing all utilities \n"
    all_utilities = Dict{Vector{Int64},Float64}()
    Threads.@threads for S in v.all_subsets
        all_utilities[S] = utility(v, S, b)
    end
    shapley_values = Vector{Float64}()
    for i in 1:num_clients
        @printf "start computing shapley value for client %i \n" i
        # power set of [M] \ {i}
        all_subsets_i = collect( powerset(setdiff(collect(1:num_clients), [i]), 1) )
        si = 0.0
        for S in all_subsets_i
            c = 1 / binomial(num_clients - 1, length(S))
            # U(S ∪ i)
            u1 = all_utilities[ sort(vcat(S, [i])) ]
            # U(S)
            u2 = all_utilities[ S ]
            si += c*(u1 - u2)
        end
        push!(shapley_values, si)
    end
    return shapley_values
end

# compute shapley value via monte carlo
function compute_shapley_value_monte_carlo(v::AsynValuator, b::Vector{Float64})
    @printf "start computing Shapley value with Monte Carlo \n"
    num_clients = v.num_clients 
    K = 10*num_clients
    perm = collect(1:num_clients)
    shapley_values = zeros(num_clients)
    Threads.@threads for k = 1:K
        @printf "permutation %d \n" k
        # generate random permutation
        randperm!(perm)
        # previous utility
        pre = 0.0
        # add utilities to shapley values
        for i = 1:num_clients
            idx = perm[i]
            curr = utility(v, perm[1:i], b)
            shapley_values[idx] += (curr - pre) / K
            pre = curr
        end
    end
    return shapley_values
end