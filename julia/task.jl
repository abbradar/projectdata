using DataFrames
using StatsBase
using MLBase
import XGBoost

include("Lib.jl")
using Lib

srand(42)

fullset = begin
    set = readtable("train.csv")
    set[2:end]
end

applycol!(x -> x - minimum(x) + 1, fullset)

# Seemingly arbitrary ¯\_(ツ)_/¯
const logscale_border = 1e4
applycol!(x -> maximum(x) > logscale_border ? log(x) : x, fullset)

applycol!(uninormalize, fullset)

trainset[end] = map(Int, trainset[end])

(trainset, testset) = train_test_split(fullset, test_size=0.25)
train_n = size(trainset)[1]

# First and, well, the best idea. Oh well...
function est_model(ind)
    nrounds = 1
    XGBoost.xgboost(convert(Matrix, trainset[ind, 1:end-1]), nrounds, label=convert(Array, trainset[ind, end]), max_depth=3, n_estimators=300, learning_rate=0.05)
end

function eval_model(model, ind)
    r = XGBoost.predict(model, convert(Matrix, trainset[ind, 1:end-1]))
    pts = roc(convert(Array, trainset[ind, end]), r, 0:step:1)
    
end

scores = cross_validate(est_model, eval_model, train_n, Kfold(train_n, 5))
