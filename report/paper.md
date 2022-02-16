# Introduction

The advent of computers came with the ability of performing 
calculations in a seamless manner, and brought with it the 
possibility of optimizing mathematical functions by finding its optimum set 
of values, this process is referred to as numerical optimization.

The realm of numerical optimization is vast and contains many algorithms whose
inspiration can be traced back to the display of populations and the basic 
social rules that govern them. A particular family of algorithms are those 
that mimic the behaviour of insects that as a collective they appear to be 
intelligent, but when separated the individual agent does not appear to be 
autonomous, the organisms that fit this definition are referred to as 
swarm intelligence (SI).

In this work an SI algorithm known as Artificial Bee Colony (ABC) was chosen, 
which simulates the way in which honey bees split their work for foraging and
securing food.

The novelty of this work is that of implementing the aforementioned algorithm
in a GPU.

# Background
## Artificial Bee Colony

The Artificial Bee Colony algorithm [@abc-k] simulates the behaviour of
bees whose main goal is to gather food. The different bees involved in this
are:

- **Employed bee**: Bee that has found a food source and determined how 
	good it is, it then goes to the hive to relay the food source's 
	location to other bees through a "waggle dance".[@bee-dance]
- **Onlooker bee**: Observes the employed bee's dance and determines if the 
	food source is good enough.
- **Scout bee**: Once the current source has been depleted the bee is set to 
forage for other food sources, thus, it creates a random solution and, if
its fitness is better than that of the previous solution the bee's task is 
set to employed, otherwise it changes to an onlooker.

The foundation of this algorithm lies in the exchange of information between 
foraging bees, this is were the intelligent behaviour is observed.

## Implementation

Every single bee has an index $i \in [0,N-1] \cap \mathbb{N}$ that 
uniquely identifies them; $N = |SB| + |EB| + |OB|$ which 
relays that the total number of bees equals the sum of the amount of 
scout bees, employed bees and onlooker bees.
($SB$, $EB$ and $OB$ are vectors that contain the indices of scouting, 
employed and onlooker bees respectively).

The probability of any given employed bee to be chosen during the onlooker
phase is:

$$ P_{i} = \frac{fit_{i}}{\sum_{j \in EB}fit_{j}} $$ {#eq:probFitness}

Where $i \in EB$.

The way in which new potential solutions are computed can be seen in the
equation below.

$$ v_{ij} = x_{ij} + \phi(x_{ij} - x_{kj}) $$ {#eq:randomvar}

Where $i \in EB \cup OB$, $k \in EB$ and $\phi$ is a random number between 0 and 1.

There's a constraint however, $i \neq k$ (Otherwise the position would not be updated).

0. Define the function to maximize and the amount of cycles to run for.
1. Initialize the bee's solutions.
2. Compute every bee's fitness.
3. The employed bee selects a random employed bee (With equal probability on 
selecting any of them), computes a potential solution (As seen in [@eq:randomvar] 
and its fitness, and, if it performs better, the new solution is adopted.
4. The onlooker bee selects a random employed bee (Giving preference to the 
bees with the best fitnesses as seen in [@eq:probFitness]), computes a 
potential solution (As seen in [@eq:randomvar] and its fitness, 
and, if it performs better, the new solution is adopted.
5. In the event that a bee has not improved its solution for a given amount of
cycles, the bee is turned into a scout bee, which sets its solution wiith random
values.
6. Repeat from 3 until the cycles end.

## Parallel implementation

In this parallel implementation every thread is interpreted to be a bee and every 
block a hive, the choices during the code's development are discussed in the 
following points:

### Setting a bee's task

Knowing that every thread is a bee, they still have to be identified somehow, 
the proposed solution is that of using an enumerator type and storing them 
in a shared memory.

### Obtaining random numbers

There were two possibilites:

* Creating and moving into the device's global memory an array filled with 
pre-computed random values.
* Computing the random values directly in the device.

Due to the large amount of data that had to be stored in the first case
the second option was chosen out of convenience.

### Re-assigning task to a scout bee

A bee's scouting task will only last for a single iteration, after which its
job will change depending on the solution's fitness, in the event it's better
the bee will become employed, otherwise it will become an onlooker.

### Multihive computation

As previously stated, a block inside of the GPU corresponds to a hive [@hive], 
operating independently from other hives. Their independence is largely
due to the impossibility for blocks to exchange information between each
other in a synchronized manner.

### Crowd wisdom

It's the idea that aggregating the results from many independent sources
will result in a consensus closer to the truth [@crowd-wisdom], 
this is theorized to be due to the fact that independent actors 
are susceptible to errors, and such error can be eliminated through the 
correct aggregation of the multitude of data. 

## Code implementation

### Obtaining random numbers

The library used for obtaining random numbers is cuRAND, which requires
its initialization and a seed to instantiate a state, which will then 
be used to get a stochastic number.

### Error detection

A library already present in CUDA's examples was used (`cuda_helper.h`), which
checks the return values of various CUDA standard API.

## Aggregating the results

After the hives return their final fitnesses they are aggregated 
in the following way:

$$ sol_{weighted_{i}} = \sum_{j=0}^{HN}(sol_{ij} \cdot \frac{fit_{j}}{fit_{best}}) $$ {#eq:finalFitness}

Where HN is the number of hives that were instantiated.

### Managing potential overflow

There exists a latent possibility for overflow to occur during the fi manaagement 

The latent possibility of existing overflow due to managing fitnesses which are

# Methodology

The application was developed using NVIDIA SDK's tools 
(cuda-gdb for debugging, cuda-memcheck for checking memory, among others)

## Benchmark functions

For testing the algorithm's correcteness the following benchmark functions 
were implemented:

### Rastrigin

$$ f(\bold{x}) = 10n + \sum_{i=1}^{n}(x_{i}^{2} - 10cos(2 \pi x_{i})) $$ {#eq:rastrigin}

Where $\bold{x} \in \mathbb{R}^{n}$.
The global minimum is $\bold{x} = [0, 0, ..., 0]$.

The contour and surface for a bidimensional Rastrigin can be seen in [@fig:rastriginContour] and [@fig:rastriginSurface].

### Sphere

$$ f(\bold{x}) = \sum_{i=1}^{n}{x_{i}^{2}} $$ {#eq:sphere}

Where $\bold{x} \in \mathbb{R}^{n}$.
The global minimum is $\bold{x} = [0, 0, ..., 0]$.

The surface for a bidimensional spheric function can be seen in [@fig:sphereSurface].

### Rosenbrock

$$ f(\bold{x}) = \sum_{i=1}^{n-1}(100(x_{i+1} - x_{i}^{2})^{2} + (1 - x_{i}^{2}))$$ {#eq:rosenbrock}

Where $\bold{x} \in \mathbb{R}^{n}$.
The global minimum is $\bold{x} = [1, 1, ..., 1]$.

The surface for a bidimensional Rosenbrock can be seen in [@fig:rosenbrockSurface].

## Mapping the functions

Due to the benchmark functions being a minimization problem they are mapped
into a maximization function by applying:

$$ fitness_{max} = \frac{1}{fitness_{min}} $$ {#eq:min2max}

The behaviour of this mapping applied to the Rosenbrock function can be seen in [@fig:invRosenbrockContour] and [@fig:inbRosenbrockSurface].

## Error measurement

Measured in mean squared error.

$$ MSE = \frac{1}{n}\sum_{i=1}^{n}(x_{i} - \hat{x_{i}}) $$ {#eq:mse}

Where n is the dimension of the solution.

# Results

The baseline implementation has the following parameters:

* 8 blocks.
* 128 threads.
* 3 maximum patience.
* -3.0 minimum float.
* 3.0 maximum float.
* 0.5 onlooker to employed ratio.
* 0 seed for initializing the random states.
* 2 solution dimension.

## Performance - timing

| Iterations | Rastrigin | Spheric | Rosenbrock |
| ---------- | --------- | ------- | ---------- |
| 8    | 3.2 ms | 3 ms | 3.03 ms |
| 64   | 24.2 ms | 22 ms | 23.74 ms |
| 256  | 96.83 ms | 87 ms | 94.30 ms |
| 1024 | 387.38 ms | 351.45 ms | 377.02 ms |

: Amount of iterations vs the main kernel execution time. {#tbl:iterVStiming}

## Performance - error

| Iterations | Rastrigin | Spheric | Rosenbrock |
| ---------- | --------- | ------- | ---------- |
| 8    | ($0.106$, $8.13 \cdot 10^{-2}$)    | ($1.3 \cdot 10^{-2}$, $-5.87 \cdot 10^{-3}$) | (0.939,0.906) |
| 64   | ($2.24 \cdot 10^{-2}$, $1.86 \cdot 10^{-3}$)  | ($1.97 \cdot 10^{-3}$, $5.63 \cdot 10^{-3}$) | (1.011,1.022) |
| 256  | ($-1.82 \cdot 10^{-3}$, $2.88 \cdot 10^{-3}$) | ($1.52 \cdot 10^{-3}$, $-8.16 \cdot 10^{-4}$) | (1.001018,1.002216) |
| 1024 | ($-3.3 \cdot 10^{-4}$, $5.83 \cdot 10^{-3}$)  | ($9.76 \cdot 10^{-4}$, $4.19 \cdot 10^{-4}$) | (1.000574,1.000933) |

: Amount of iterations versus the obtained result. {#tbl:iterVSres}

| Iterations | Rastrigin | Spheric | Rosenbrock |
| ---------- | --------- | ------- | ---------- |
| 8    | $9 \cdot 10^{-3}$ | $1.1 \cdot 10^{-4}$  | $6.15 \cdot 10^{-3}$ |
| 64   | $2 \cdot 10^{-4}$ | $1.77 \cdot 10^{-5}$ | $3.24 \cdot 10^{-4}$ |
| 256  | $5.81 \cdot 10^{-5}$ | $1.48 \cdot 10^{-6}$ | $2.97 \cdot 10^{-6}$ |
| 1024 | $1.70 \cdot 10^{-5}$ | $5.64 \cdot 10^{-7}$ | $5.99 \cdot 10^{-7}$ |

: Amount of iterations versus error. {#tbl:iterVSerror}

| Threads | Rastrigin | Spheric | Rosenbrock | 
| ------- | --------- | ------- | ---------- |
| 32 | $7.4 \cdot 10^{-3}$ | $1.1 \cdot 10^{-3}$ | $1.5 \cdot 10^{-5}$ |
| 64 | $2.32 \cdot 10^{-6}$ | $4.01 \cdot 10^{-4}$ | $7.6 \cdot 10^{-6}$ |
| 128 | $5.93 \cdot 10^{-6}$ | $2.19 \cdot 10^{-6}$ | $1.03 \cdot 10^{-6}$ |

: Amount of threads vs error. {#tbl:threadsVSerror}

# Discussion

The main factor that affects the amount of time the kernel is running is the
amount of iterations, this effect can be seen in [@tbl:iterVStiming].

## Correctness of the algorithm

The bees in a hive are correctly gathering around optimum points, as can 
be seen in [@fig:rastriginSol8].

Given the algorithm is running for long enough the solution will converge
in the global minimum, as can be seen between [@fig:rastriginSol8] and 
[@fig:rastriginSol256].

The algorithm succesfully managed to find the global optimum as can be seen
in [@fig:rastriginSol256], [@fig:sphericSol256] and [@fig:rosenbrockSol256].

In summary, for obtaining the best results there needs to be a combination of:

* Large enough amount of hives (More than 4 blocks).
* Large enough amount of bees (More than 32 threads).
* Large enough amount of iterations (More than 32 iterations).

# Ablation

## Storing the passed struct in constant memory

There was no noticeable difference in performance.

## Using shared memory for storing fitness information

It consistently performs 1.2 times longer than its global memory counterpart, 
the reason for this is still unknown, considering that the shared memory 
is being accessed in a sequential mode, which entails that there will not
be any bank conflict.

\newpage

# Figures

![Contour of the Rastrigin function](../images/rastrigin_contour.png){#fig:rastriginContour width=75%}

![Surface of the Rastrigin function](../images/rastrigin_surface.png){#fig:rastriginSurface width=75%}

![Surface of the spheric function](../images/sphere_surface.png){#fig:sphereSurface width=75%}

![Surface of the Rosenbrock function](../images/rosenbrock_surface.png){#fig:rosenbrockSurface width=75%}

![Contour of the inverse Rosenbrock function](../images/inverse_rosenbrock_contour.png){#fig:invRosenbrockContour width=75%}

![Surface of the inverse Rosenbrock function](../images/inverse_rosenbrock_surface.png){#fig:inbRosenbrockSurface width=75%}

![Solutions of Rastrigin after 8 iterations](../images/rastrigin_8_05_3.png){#fig:rastriginSol8}

![Solutions of Rastrigin after 256 iterations](../images/rastrigin_256_05_3.png){#fig:rastriginSol256}

![Solutions of the spheric function after 8 iterations](../images/spheric_8_05_3.png){#fig:sphericSol8}

![Solutions of the spheric function after 256 iterations](../images/spheric_256_05_3.png){#fig:sphericSol256}

![Solutions of Rosenbrock function after 256 iterations](../images/rosenbrock_256_05_3.png){#fig:rosenbrockSol256}

\newpage

# Conclusion

Through the usage of GPUs lots of computations can be performed in parallel, 
effectively accelerating the computational time. One of the potential 
applications are relayed in this study.

The Artificial Bee Colony algorithm is found to be a prime candidate for 
exploiting the Single Instruction Multiple Data model due to the need of
large amount of computations to be performed.

# Future work

It's recommended to study further optimizations on the algorithm's workflow.

# Further comments

## Compilation step

In order to avoid the creation of a monolithic file that would contain 
all of the functions, the compilation is performed in a detached mode 
(Option -dc is added to the compiler's flags), which means that
the compiler will create relocatable device code, which will then be linked
in a future step once all of the object files are generated.

## Debugging code

The intermediate files created during the compilation and linking of
the program are required for understanding the mapping between the 
machine code and source code, this is of utmost importance for using
the cuda-gdb and cuda-memcheck tools. For succesfully doing so
the flags -g, -G and -keep have to be set both during the 
generation of object files and the linking stage.

Another factor that affected the correct debugging of the program is
the fact that, by default, the created user has no permissions to access the 
debugging intermediary device (Such device can be found in 
`/dev/nvhost-dbg-gpu`).

## Disabling GPU timeout

Whenever the profiler was used to obtain information about the code 
it would take longer than regular execution, this is due to the fact
that there's an associated overhead produced by it to obtain
and extract the statistics at strategic points. Many online resources 
pointed that the solution lied in disabling the Graphical
User Interface (Uninstalling X and setting the execution level to multiuser), 
however this would not solve the issue.
The solution instead was found by modifying the device's settings manually 
(Setting `/sys/kernel/debug/gpu.0/timeouts_enabled` to N).
After manually disabling the setting, if the device's properties are accessed
(Through `cudaGetDeviceProperties()` -> `kernelExecTimeoutEnabled`) it gives 
a misleading result, claiming that execution timeout is still enabled.

## Issues with nvprof

Whenever the profiling had to be performed on a program that featured shared
fitness memory, the nvprof utility would not be able to execute or gather 
metrics/events, citing a `cudaErrorLaunchFailure` as its root cause, 
despite the fact that the program ran succesfully without the usage of nvprof. 
Neither cuda-gdb nor cuda-memcheck were useful for pinpointing the issue. 
statistics correctly. No solution to this issue was found.

# References


