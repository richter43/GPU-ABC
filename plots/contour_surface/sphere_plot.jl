using Plots
sphere(x) = sum(x.^2)

dpi = 600
x = y = range(-3, stop=3, length=200)

l = Vector{Float64}()
for i in x
   for j in y
       v = [i j]
       append!(l, sphere(v))
   end 
end

surface(x,y,l, dpi=dpi)
png("./sphere_surface.png")
contour(x,y,l, dpi=dpi, fill = true)
png("./sphere_contour.png")
