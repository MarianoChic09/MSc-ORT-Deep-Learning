# using Pkg
# Pkg.add("Flux")

using CUDA

if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

using Flux
using Flux: params

# using Flux: @epochs

epochs = 5000

# Datos de entrenamiento
X = Float32[0 0; 0 1; 1 0; 1 1]'

# Etiquetas combinadas (AND, OR, XOR, NOR, NAND)
y_combined = Float32[0 0 0 1 1; 0 1 1 0 1; 0 1 1 0 1; 1 1 0 0 0]'

# Definición del modelo
model = Chain(
    Dense(2, 10, σ),  # Capa oculta con 10 neuronas
    Dense(10, 5, σ)   # Capa de salida con 5 neuronas (AND, OR, XOR, NOR, NAND)
)

# Función de pérdida
loss(x, y) = Flux.Losses.logitbinarycrossentropy(model(x), y)

# Optimizador
opt = ADAM()

# Entrenamiento del modelo
for epoch in 1:epochs
    if epoch % 1000 == 0
        @info "Epoch $epoch"
    end
    Flux.train!(loss, params(model), [(X, y_combined)], opt)
end

# Predicciones
predictions = model(X)
println(round.(predictions))
