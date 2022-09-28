########################################################################
# Client for synchronous learning
########################################################################
mutable struct Client{T1<:Int64, T2<:Float64, T3<:Vector{T1}, T4<:Matrix{T2}, T5<:SparseMatrixCSC{T2, T1}}
    id::T1                                  # client index
    Xtrain::T5                              # training data
    Xtest::T5                               # test data
    num_classes::T1                         # number of classes
    num_clients::T1                         # number of clients
    num_epoches::T1                         # number of epoches
    batch_size::T1                          # number of batches
    learning_rate::T2                       # learning rate
    W::T4                                   # client model
    batch::T3                               # mini-batch
    grads::T4                               # gradient information
    λ::T2                                   # regularization parameter
    function Client(id::Int64, Xtrain::SparseMatrixCSC{Float64, Int64}, Xtest::SparseMatrixCSC{Float64, Int64}, config::Dict{String, Union{Int64, Float64}}; λ::Float64 = 0.0)
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        num_epoches = config["num_epoches"]
        batch_size = config["batch_size"]
        learning_rate = config["learning_rate"]
        dm = size(Xtrain, 1)
        W = zeros(Float64, num_classes, dm)
        batch = zeros(Int64, batch_size)
        grads = zeros(Float64, num_classes, batch_size)
        new{Int64, Float64, Vector{Int64}, Matrix{Float64}, SparseMatrixCSC{Float64, Int64}}(id, Xtrain, Xtest, num_classes, num_clients, num_epoches, batch_size, learning_rate, W, batch, grads, λ)
    end
end

# update batch 
function update_batch(c::Client, batch::Vector{Int64})
    c.batch .= batch
end

# update gradient information
function update_grads(c::Client, grads::Matrix{Float64})
    c.grads .= grads
end

# update client model W
function update_model(c::Client)
    Xbatch = c.Xtrain[:, c.batch]
    Wgrad = (c.grads * Xbatch') ./ c.batch_size
    c.W .*= (1.0 - c.learning_rate*c.λ)
    c.W .-= c.learning_rate * Wgrad 
end

########################################################################
# Client for asynchronous learning
########################################################################
mutable struct AsynClient{T1<:Int64, T2<:Float64, T3<:Matrix{T2}, T4<:SparseMatrixCSC{T2, T1}}
    id::T1                                  # client index
    Xtrain::T4                              # training data
    Xtest::T4                               # test data
    num_classes::T1                         # number of classes
    num_clients::T1                         # number of clients
    batch_size::T1                          # number of batches
    learning_rate::T2                       # learning rate
    W::T3                                   # client model
    grads::T3                               # gradient information
    ts::T2                                  # time gap between successive communications
    num_commu::T1                           # number of communication rounds
    λ::T2                                   # regularization parameter
    function AsynClient(id::Int64, Xtrain::SparseMatrixCSC{Float64, Int64}, Xtest::SparseMatrixCSC{Float64, Int64}, ts::Float64, config::Dict{String, Union{Int64, Float64}}; λ::Float64 = 0.0)
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        batch_size = config["batch_size"]
        learning_rate = config["learning_rate"]
        dm = size(Xtrain, 1)
        W = zeros(Float64, num_classes, dm)
        grads = zeros(Float64, num_classes, batch_size )
        num_commu = 0
        new{Int64, Float64, Matrix{Float64}, SparseMatrixCSC{Float64, Int64}}(id, Xtrain, Xtest, num_classes, num_clients, batch_size, learning_rate, W, grads, ts, num_commu, λ)
    end
end

function update_grads(c::AsynClient, grads::Matrix{Float64})
    @printf "Client %i finish downloading gradient \n" c.id
    c.num_commu += 1
    c.grads = grads
end

function update_model(c::AsynClient, batch::Vector{Int64})
    Xbatch = c.Xtrain[:, batch]
    Wgrad = (c.grads * Xbatch') ./ c.batch_size
    c.W .*= (1.0 - c.learning_rate*c.λ)
    c.W .-= c.learning_rate * Wgrad 
end

