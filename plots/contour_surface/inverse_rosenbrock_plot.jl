using Plots
rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

linspace(a,b,N)= (a+ (b-a)*(i)/(N-1) for i=0:(N-1))

x = y = broadcast(exp,collect(linspace(log(0.5), log(1.5), 100)))



dpi = 600

l = Vector{Float64}()
for i in x
   for j in y
       v = [i j]
       append!(l, 1/rosenbrock(v))
   end 
end

surface(x,y,l, dpi=dpi)
png("./inverse_rosenbrock_surface.png")
contour(x,y,l, dpi=dpi)
png("./inverse_rosenbrock_contour.png")
