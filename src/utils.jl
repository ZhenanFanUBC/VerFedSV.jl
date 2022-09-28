########################################################################
# Helper Functions
########################################################################

# softmax function
function softmax(z::Vector{Float64})
    expz = exp.(z)
    s = sum(expz)
    return expz ./ s
end

# negative log-likelihood function
function neg_log_loss(z::Vector{Float64}, y::Int64)
    return -log(z[y])
end

# generate random batches
function generate_batches(num_data::Int64, num_batches::Int64)
    n,r = divrem(num_data, num_batches)
    b = collect(1:n:num_data+1)
    for i in 1:length(b)
        b[i] += i > r ? r : i-1  
    end
    p = randperm(num_data)
    return [p[r] for r in [b[i]:b[i+1]-1 for i=1:num_batches]]
end

# vertically split data
function split_data(Xtrain::SparseMatrixCSC{Float64, Int64}, Xtest::SparseMatrixCSC{Float64, Int64}, num_clients::Int64)
    num_features = size(Xtrain, 1)
    num_features_client = div(num_features, num_clients)
    Xtrain_split = Vector{ SparseMatrixCSC{Float64, Int64} }(undef, num_clients)
    Xtest_split = Vector{ SparseMatrixCSC{Float64, Int64} }(undef, num_clients)
    t = 1
    for i = 1:num_clients
        if i < num_clients
            ids = collect(t: t+num_features_client-1)
        else
            ids = collect(t: num_features)
        end
        Xtrain_split[i] = Xtrain[ids, :]
        Xtest_split[i] = Xtest[ids, :]
        t += num_features_client
    end
    return Xtrain_split, Xtest_split
end

# load data 
function load_data(filename::String)
    if filename == "mnist"
        fid = h5open("./data/MNIST/MNISTdata.hdf5", "r")
        data = read(fid)
        close(fid)
        Xtrain = convert(Matrix{Float64}, data["x_train"]); Xtrain = sparse(Xtrain)
        Ytrain = convert(Matrix{Int64}, data["y_train"]); Ytrain = Ytrain[:]; Ytrain .+= 1
        Xtest = convert(Matrix{Float64}, data["x_test"]); Xtest = sparse(Xtest)
        Ytest = convert(Matrix{Int64}, data["y_test"]); Ytest = Ytest[:]; Ytest .+= 1
        return Xtrain, Ytrain, Xtest, Ytest
    elseif filename == "adult"
        Xtrain, Ytrain = read_libsvm("./data/Adult/a8a"); Xtrain = Xtrain[1:end-1, :]
        Xtest, Ytest = read_libsvm("./data/Adult/a8a.t")
        return Xtrain, Ytrain, Xtest, Ytest
    elseif filename == "rcv1"
        X, Y = read_libsvm("./data/RCV1/rcv1_train.binary")
        N = 15000
        perm = randperm(length(Y))
        train_idx = perm[1:N]; test_idx = perm[N+1:end]
        Xtrain, Ytrain = X[:,train_idx], Y[train_idx]
        Xtest, Ytest = X[:,test_idx], Y[test_idx]
        return Xtrain, Ytrain, Xtest, Ytest
    elseif filename == "covtype"
        X, Y = read_libsvm("./data/Covtype/covtype.scale01")
        N = 100000
        perm = randperm(length(Y))
        train_idx = perm[1:N]; test_idx = perm[N+1:end]
        Xtrain, Ytrain = X[:,train_idx], Y[train_idx]
        Xtest, Ytest = X[:,test_idx], Y[test_idx]
        return Xtrain, Ytrain, Xtest, Ytest
    elseif filename == "web"
        Xtrain, Ytrain = read_libsvm("./data/Web/w7a")
        Xtest, Ytest = read_libsvm("./data/Web/w7a.t")
        return Xtrain, Ytrain, Xtest, Ytest
    else
        @printf "Unsupported filename"
    end
end

# read data from libsvm
function read_libsvm(filename::String)
    numLine = 0
    nnz = 0
    open(filename, "r") do f
        while !eof(f)
            line = readline(f)
            info = split(line, " ")
            numLine += 1
            nnz += ( length(info)-1 )
            if line[end] == ' '
                nnz -= 1
            end
        end
    end
    @printf("number of lines: %i\n", numLine)
    n = numLine
    m = 0
    I = zeros(Int64, nnz)
    J = zeros(Int64, nnz)
    V = zeros(Float64, nnz)
    y = zeros(Int64, n)
    numLine = 0
    cc = 1
    open(filename, "r") do f
        while !eof(f)
            numLine += 1
            line = readline(f)
            info = split(line, " ")
            value = parse(Int64, info[1] )
            if value < 0
                value = Int64(2)
            end
            y[numLine] = value
            ll = length(info)
            if line[end] == ' '
                ll -= 1
            end
            for i = 2:ll
                idx, value = split(info[i], ":")
                idx = parse(Int, idx)
                value = parse(Float64, value)
                I[cc] = numLine
                J[cc] = idx
                V[cc] = value
                cc += 1
                m = max(m, idx)
            end
        end
    end
    return sparse( J, I, V, m, n ), y
end

# matrix completion
function complete_matrix(A::SparseMatrixCSC{Float64}, r::Int64)
    I, J, ~ = findnz(A)
    nnz = length(I)
    obs = [(I[k], J[k]) for k = 1:nnz]
    loss = QuadLoss()
    reg = QuadReg(.1)
    glrm = GLRM(A, loss, reg, reg, r, obs=obs)
    X, Y, ch = fit!(glrm)
    @printf "finish matrix completion"
    return convert(typeof(Y), X'), Y
end

# update embedding matrix
function update_embedding_matrix!(H::SparseMatrixCSC{Float64, Int64}, batch::Vector{Int64}, embeddings::Matrix{Float64}, t::Int64)
    num_classes = size(embeddings, 1)
    batch_size = length(batch)
    @inbounds for i = 1:num_classes
        @inbounds for j = 1:batch_size
            H[batch[j], (t-1)*num_classes+i] = embeddings[i, j]
        end
    end
    return nothing
end

