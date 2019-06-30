Sources for the implementations for the paper 

Giovanni Mahlknecht, Anton Dignös, Johann Gamper: 
A scalable dynamic programming scheme for the computation of 
optimal k-segments for ordered data. Inf. Syst. 70: 2-17 (2017)


#### Compile instructions 

make targets
all: make all runtimes - destination directory=./binrt
debug: make the debug runtimes: destination directory=./bindebug
statistics [shows additionl informations, i.e. graph size, number of computations etc.]: destination directory=./binstat


Different executables are generated out of the sources. The compile process is controlled through compiler switches:
EARLYBREAK: computation is stopped as soon as the error limit is reached
DP: use diagonal pruning
GP: use gap pruning
BINSEARCH: use binary search for retrieval of the start point of the computation
SEEDJAG: use Jagadish' seed for binsearch. Only in combination with BINSEARCH.
SEEDOPTPREV: use optimum previous as seed for binsearch. Only in combination with BINSEARCH.
SEEDOUR: use as seed upper row j-delta to j. Only in combination with BINSEARCH.
GRAPH: use split point graph if not set use matrix
NODEPOOL: use node pooling instead of alocation/deallocation only in combination with GRAPH

Executables and the used compile switches
ptamatnaive: EARLYBREAK
ptamat: DEARLYBREAK, GAPPRUNING
ptamatdp: DEARLYBREAK, GAPPRUNING, DP
ptamatdpnogaps: EARLYBREAK DP
ptamatjag: EARLYBREAK BINSEARCH SEEDJAG
ptamatjagplus: EARLYBREAK BINSEARCH SEEDOPTPREV
ptamatour: EARLYBREAK DP GAPPRUNING BINSEARCH SEEDOUR
ptagraph: EARLYBREAK DP GAPPRUNING GRAPH NODEPOOL
ptagraphjag: EARLYBREAK GRAPH NODEPOOL BINSEARCH SEEDJAG
ptagraphjagplus: EARLYBREAK GRAPH NODEPOOL BINSEARCH SEEDOPTPREV
ptagraphour: EARLYBREAK DP GAPPRUNING GRAPH NODEPOOL BINSEARCH SEEDOUR

Parallelized implementations - same configurations as singlecore but uses parallel threads 
ptamultimat: EARLYBREAK GAPPRUNING 
ptamultimatdp: EARLYBREAK DP GAPPRUNING 
ptamultigraph: EARLYBREAK DP GAPPRUNING GRAPH NODEPOOL 
ptamultimatjag: EARLYBREAK GAPPRUNING BINSEARCH SEEDJAG 
ptamultimatjagplus: EARLYBREAK GAPPRUNING BINSEARCH SEEDOPTPREV 
ptamultimatour: EARLYBREAK DP GAPPRUNING BINSEARCH SEEDOUR 
ptamultigraphjag: EARLYBREAK GAPPRUNING GRAPH DNODEPOOL BINSEARCH SEEDJAG 
ptamultigraphjagplus EARLYBREAK GAPPRUNING GRAPH DNODEPOOL BINSEARCH SEEDOPTPREV 
ptamultigraphour EARLYBREAK DP GAPPRUNING GRAPH DNODEPOOL BINSEARCH SEEDOUR

oksm: Optimum k Segments with graph implementation, parallelized implementation and all optimizations as described in the paper



#### usage
all executeables have the same parameters
-i <dataset>  retrieves the dataset from postgres db. It accesses the table (or view) with the name <dataset> with schema (quantity: double precision)
   if -i switch is not set random data is generated
-n <num> size of tuples to read (or generate)
-c <num> size of the reduction (buckets) 
-o output of the the resulting splitpath on screen


