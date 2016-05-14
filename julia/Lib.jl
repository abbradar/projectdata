module Lib

export uninormalize, applycol!, train_test_split

using DataFrames
using StatsBase

function uninormalize(v)
    min = minimum(v)
    max = maximum(v)
    if (min == max)
        fill!(copy(v), 0)
    else
        (v - min) / (max - min)
    end
end

function applycol!(f :: Function, df :: DataFrame)
    for (name, column) in eachcol(df)
        df[name] = f(column)
    end
    df
end

function train_test_split(data...; test_size=nothing, train_size=nothing, dim=1)
    if test_size != nothing && train_size != nothing
        error("Must not provide both test and train sizes")
    elseif test_size == nothing
        if train_size == nothing
            test_size = 0.25
        else
            test_size = 1 - train_size
        end
    end
    if test_size < 0 test_size > 1
        error("Test and train sizes must be between 0 and 1")
    end

    data_n = size(data[1])[dim]
    for i in data
        if size(i)[dim] != data_n
            error("Sizes of input arrays differ")
        end
    end

    test_n = Integer(round(data_n * test_size))
    train_n = data_n - test_n

    s = sample(1:data_n, test_n, replace=false)
    not_s = collect(setdiff!(Set(1:data_n), s))

    test_dest = map(data) do arr
        test_slice = fill!(cell(ndims(arr)), :)
        test_slice[dim] = s
        arr[test_slice...]
    end

    train_dest = map(data) do arr
        train_slice = fill!(cell(ndims(arr)), :)
        train_slice[dim] = not_s
        arr[train_slice...]
    end

    if length(data) == 1
        (train_dest[1], test_dest[1])
    else
        (train_dest, test_dest)
    end
end

end
