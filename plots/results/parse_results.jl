using Plots
using CSV
using DataFrames
using ArgParse

function parse_commandline()

	s = ArgParseSettings()

	@add_arg_table s begin
		"--function"
		required = true
		"--file"
		required = true
		"--dpi"
		arg_type = Int
		default = 600
	end
	
	return parse_args(s)
end

function get_fitness_function(args)
	
	if cmp(args["function"], "rastrigin") == 0
		fun = x -> 10length(x) + sum(x.^2 .- 10 .* cos.(2π .* x))
	elseif cmp(args["function"], "spheric") == 0
		fun = x -> sum(x.^2)
	elseif cmp(args["function"], "rosenbrock") == 0
		fun = x -> (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2 
	end
	
	return fun
end

function main()	
	args = parse_commandline()
	x = y = range(-0.5, stop=0.5, length=200)

	fun = get_fitness_function(args)

	l = Vector{Float64}()
	for i in x
	   for j in y
	       v = [i j]
	       append!(l, fun(v))
	   end
	end

	csv_file = CSV.read(args["file"], DataFrame)
	filename = split(args["file"], "/")[2]

	contour(x,y,l, dpi=args["dpi"])
	for (i, idx) in enumerate(csv_file.hive)
		if idx == -1
			label="WoC"
		else
			label="Hive $idx"
		end
		scatter!([csv_file.x[i]], [csv_file.y[i]], label=label)
	end
	png("./"*args["function"]*"/"*filename*".png")

end

main()
