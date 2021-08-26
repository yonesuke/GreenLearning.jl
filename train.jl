using Flux
using Flux.Data: DataLoader
using BSON: @save, @load # for data saving
using ProgressMeter
using LinearAlgebra
using Dates

pdename = ARGS[1]
@info "Learning Green function of $pdename equation"

cp = "checkpoints_$(pdename)_$(now())"
mkdir(cp)
@info "$cp folder created"

# Load mat file
using MAT
file = matopen("greenlearning/examples/datasets/$(pdename).mat")
varnames = collect(names(file))
for varname in varnames
    val = read(file, varname)
    @eval $(Symbol(varname)) = $val
end
close(file)
X = reshape(X, 100)
Y = reshape(Y, 200)
xmin, xmax = X[begin], X[end]
ymin, ymax = Y[begin], Y[end]
dx = X[2]-X[1]
dy = Y[2]-Y[1]
Nu, Nf = length(X), length(Y)
@info "Train data loaded"

hidden_size = 50
G_network = Chain(
    Dense(2, hidden_size, relu),
    Dense(hidden_size, hidden_size, relu),
    Dense(hidden_size, hidden_size, relu),
    Dense(hidden_size, hidden_size, relu),
    Dense(hidden_size, 1)
)
U_hom_network = Chain(
    Dense(1, hidden_size, relu),
    Dense(hidden_size, hidden_size, relu),
    Dense(hidden_size, hidden_size, relu),
    Dense(hidden_size, hidden_size, relu),
    Dense(hidden_size, 1)
)
parameters = Flux.params(G_network, U_hom_network)
@info "Model" G_network U_hom_network

# integrator weights
trapezoidal_x = ones(Nu); trapezoidal_x[[begin,end]] .= 0.5
trapezoidal_y = ones(Nf); trapezoidal_y[[begin,end]] .= 0.5

function loss(U_train, F_train)
    l = 0.0
    U_hom_predict = [U_hom_network([x])[1] for x in X]
    # G_predict = [G_network([x,y])[1] for x in X, y in Y]
    G_predict = [[G_network([x,y])[1] for x in X] for y in Y]
    G_predict = hcat(G_predict...)
    num = size(U_train, 2)
    for n in 1:num
        @views u = U_train[:,n] # 100×1
        @views f = F_train[:,n] # 200×1
        Gf = dy*G_predict*(trapezoidal_y.*f)
        integ = (u-U_hom_predict-Gf).^2
        loss_u = dx*dot(integ, trapezoidal_x)
        u_L2 = dx*dot(u.^2, trapezoidal_x)
        l += loss_u/u_L2
    end
    return l/num
end

opt = ADAM()
@info "Optimiser" opt

batch_size = 100
train_loader = DataLoader((U, F), batchsize=batch_size, shuffle=true)
data_num = size(U, 2)
epochs = parse(Int, ARGS[2])
digit_epochs = length(string(epochs))
iteration_num = Int(epochs*batch_size/data_num)
@info "Training start" data_num epochs batch_size iteration_num

start = now()

learning = Dict([
    ("Equation name", pdename),
    ("Total epochs", epochs),
    ("Batch size", batch_size),
    ("Number of data", data_num),
    ("Current epoch number", 0),
    ("Current loss value", []),
    ("Start time", start),
    ("Time", 0)
])

skip = 5
counter = 0
@showprogress for epoch in 1:epochs
    for (U_train, F_train) in train_loader
        gs = gradient(()->loss(U_train, F_train), parameters)
        Flux.Optimise.update!(opt, parameters, gs)
    end
    global counter += 1
    if counter % skip == 0
        d = now()
        learning["Current epoch number"] = epoch
        learning["Time"] = d
        current_loss = loss(U, F)
        push!(learning["Current loss value"], current_loss)
        fname1 = "$cp/$(pdename)_description_$(lpad(epoch, digit_epochs, "0")).bson"
        fname2 = "$cp/$(pdename)_model_$(lpad(epoch, digit_epochs, "0")).bson"
        @save fname1 learning
        @save fname2 G_network U_hom_network opt
    end
end

d = now()
learning["Current epoch number"] = epochs
learning["Time"] = d
fname1 = "$cp/$(pdename)_description_$epochs.bson"
fname2 = "$cp/$(pdename)_model_$epochs.bson"
@save fname1 learning
@save fname2 G_network U_hom_network opt
@info "Training ended\nModel saved to $fname2"
