using Plots
rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

linspace(a,b,N)= (a+ (b-a)*(i)/(N-1) for i=0:(N-1))

x = y = sort(collect(Iterators.flatten([linspace(-1,-0.1,25), linspace(0,2,1000), linspace(2.1,3,25)])))

dpi = 600

l = Vector{Float64}()
for i in x
   for j in y
       v = [i j]
       append!(l, rosenbrock(v))
   end 
end

surface(x,y,l, dpi=dpi)
png("./rosenbrock_surface.png")
contour(x,y,l, dpi=dpi)
png("./rosenbrock_contour.png")
