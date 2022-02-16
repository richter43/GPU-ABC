using Plots
rastrigin(x) = 10length(x) + sum(x.^2 .- 10 .* cos.(2π .* x))

dpi = 600
x = y = range(-3, stop=3, length=200)

l = Vector{Float64}()
for i in x
   for j in y
       v = [i j]
       append!(l, rastrigin(v))
   end 
end

surface(x,y,l, dpi=dpi)
png("./rastrigin_surface.png")
contour(x,y,l, dpi=dpi, fill = true)
png("./rastrigin_contour.png")
