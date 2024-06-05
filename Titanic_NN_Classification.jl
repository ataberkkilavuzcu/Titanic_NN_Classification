using LinearAlgebra, Zygote, ForwardDiff, Printf
using CSV, DataFrames
using StatsBase: mean
using Parameters
using Distributions
using Random
using Flux
using MLUtils
using NNlib  
using Random
using PyCall 
using Statistics
using Flux.Optimise: train!



# Load the dataset
data = CSV.read("train.csv", DataFrame)


# Preprocess the data
function preprocess_data(data)
    # Select only numeric columns
    numeric_cols = names(data, Real)
    X = data[:, numeric_cols]
    
    # Replace missing values with column mean
    for col in numeric_cols
        if any(ismissing, X[:, col])
            X[ismissing.(X[:, col]), col] = mean(skipmissing(X[:, col]))
        end
    end
    
    # Convert to Float64
    X = convert(Matrix{Float64}, Matrix(X))
    y = convert(Vector{Float64}, data[:, 2]) # Survived is the target variable
    
    # Normalize features
    X = (X .- mean(X, dims=1)) ./ std(X, dims=1)
    
    return X, y
end

X, y = preprocess_data(data)


# Split into training and test sets 80% - 20%
using Random
Random.seed!(1234)
n = size(X, 1)
indices = randperm(n)
train_indices = indices[1:round(Int, 0.8 * n)]
test_indices = indices[round(Int, 0.8 * n)+1:end]
X_train, y_train = X[train_indices, :], y[train_indices]
X_test, y_test = X[test_indices, :], y[test_indices]



# Define the model
model1 = Chain(
    Dense(size(X, 2), 64, relu),
    Dense(64, 1)
)


# Loss function
loss1(x, y) = Flux.mse(model1(x), y)

# Optimizer with learning rate 0.01
opt1 = Descent(0.01)
# Training loop
function train_model!(model, loss, opt, X_train, y_train, epochs)
    for epoch in 1:epochs
        Flux.Optimise.train!(loss, Flux.params(model), [(X_train', reshape(y_train, 1, :))], opt)
        train_loss = loss(X_train', reshape(y_train, 1, :))
        if epoch % 10 == 0
            println("Epoch: $epoch, Loss: $train_loss")
        end
    end
end

# Train the model
train_model!(model1, loss1, opt1, X_train, y_train, 100)


# Define the model
model2 = Chain(
    Dense(size(X, 2), 128, relu),
    Dense(128, 64, relu),
    Dense(64, 1)
)

# Loss function
loss2(x, y) = Flux.mse(model2(x), y)

# Optimizer with learning rate 0.001
opt2 = Descent(0.001)

# Train the model
train_model!(model2, loss2, opt2, X_train, y_train, 100)

# Define the model
model3 = Chain(
    Dense(size(X, 2), 64, relu),
    Dense(64, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 1)
)

# Loss function
loss3(x, y) = Flux.mse(model3(x), y)

# Optimizer with Adam so that we don't need to manually enter the learning rate
opt3 = Adam()

# Train the model
train_model!(model3, loss3, opt3, X_train, y_train, 100)

using Statistics

function evaluate(model, X_test, y_test)
    y_pred = model(X_test')
    for i in 1:length(y_pred)
        y_pred[i] = round(y_pred[i])
    end
    mse = mean((y_pred .- y_test).^2)
    return mse
end

# Evaluate the models
mse1 = evaluate(model1, X_test, y_test)
mse2 = evaluate(model2, X_test, y_test)
mse3 = evaluate(model3, X_test, y_test)

println("MSE for Setup 1: $mse1")
println("MSE for Setup 2: $mse2")
println("MSE for Setup 3: $mse3")

using Plots

# Function to record loss history
function train_with_history!(model, loss, opt, X_train, y_train, epochs)
    losses = []
    for epoch in 1:epochs
        Flux.Optimise.train!(loss, Flux.params(model), [(X_train', reshape(y_train, 1, :))], opt)
        push!(losses, loss(X_train', reshape(y_train, 1, :)))
    end
    return losses
end

losses1 = train_with_history!(model1, loss1, opt1, X_train, y_train, 100)
losses2 = train_with_history!(model2, loss2, opt2, X_train, y_train, 100)
losses3 = train_with_history!(model3, loss3, opt3, X_train, y_train, 100)

plot(1:100, losses1, label="Setup 1")
plot!(1:100, losses2, label="Setup 2")
plot!(1:100, losses3, label="Setup 3")
xlabel!("Epoch")
ylabel!("Loss")
title!("Learning Curves")
savefig("learning_curves.png")


