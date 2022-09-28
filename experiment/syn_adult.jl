########################################################################
# Vertical Federated Logistic Regression for Adult
########################################################################
using Printf
using SparseArrays
using Random
Random.seed!(1234)

# load data
filename = "adult"
Xtrain, Ytrain, Xtest, Ytest = load_data(filename)

# config
config = Dict{String, Union{Int64, Float64}}()
config["num_classes"] = 2
config["num_clients"] = 3
config["num_epoches"] = 20
config["batch_size"] = 2837
config["learning_rate"] = 0.2

# vertically split data
Xtrain_split, Xtest_split = split_data(Xtrain, Xtest, config["num_clients"])

# initialize server 
server = Server(Ytrain, Ytest, config)

# initialize valuator
valuator = Valuator(Ytrain, config)

# initialize clients
clients = Vector{Client}(undef, config["num_clients"])
for id = 1:config["num_clients"]
    c = Client(id, Xtrain_split[id], Xtest_split[id], config)
    clients[id] = c
    # connect with server
    connect(server, c)
end

# training
vertical_lr_train(server, clients, valuator=valuator)

# evaluation
shapley_values = evaluation(server, clients, valuator=valuator)


