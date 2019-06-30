/*  This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    */

/*
 * File: oksm.cpp
 * Optimum K Segments computation - multiple threads
 * Implementation with SPLIT POINT GRAPH and all proposed optimizations
 * NO compiler switches to control generation of different executeables
 *
 * Author: Giovanni Mahlknecht Free University of Bozen/Bolzano
 *         giovanni.mahlknecht@inf.unibz.it
 *
 * Created on March 11, 2016
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include "data.h"
#include <string.h>
using namespace std;

const double MAX = std::numeric_limits<double>::max();

std::vector<std::thread> tWorkers; //vector of threads
std::atomic<int> tWorkersBusy; //number of busy threads
std::atomic<int> tWorkersCount; //number of running threads
std::atomic<bool> tFinished; //true if computation is finished
std::atomic<bool> tStartrow; //can computation of row i start?
#ifdef STATISTICS
int maxnodes;
int numnodes;
#endif

int numthreads;
std::atomic<unsigned long> jcurr; //currently computed element
unsigned long lineJmax = 0;
unsigned long i = 0;
std::mutex multiMutex;
std::mutex graphMutex;

std::condition_variable workerHasFinished;
std::condition_variable workerStart;

itatuple* dat; //input data
unsigned long n; //dataset size
unsigned long c; //size to reduce to
struct spgnode {
  uint32_t indexnumber;
  spgnode* pathparent;
  uint32_t numchilds;
};
//structure for nodes in the graph
std::stack<spgnode*> *splitNodePool; //pool for storing unused graph nodes
spgnode **splitCur; //references to current split nodes (the level)
spgnode **splitPrev;
double* errCurr; //last two rows of error matrix
double* errPrev;
double* S; //linear sum
double* SS; //squared sum
long long* L; //L vector for weighting error tuple duration
unsigned long * G; //Gap vector
unsigned long numGaps;

/**
 * retrieve a new spgnode from the pool
 * @return a pointer to a spgnode
 */
spgnode* getNodeFromPool() {
  spgnode* newnode;
  if (splitNodePool->empty()) {
    /* create new node and return pointer to it */
    newnode = new spgnode;
  } else {
    //return from pool
    newnode = splitNodePool->top();
    splitNodePool->pop();
  }
  return (newnode);
}

/**
 * put spgnode into pool for later reutilization
 * @param node
 */
void deleteNodeToPool(spgnode* node) {
  splitNodePool->push(node);
}

/**
 *  Initialize the L, S and SS vector for error computation
 */
void initLSSSvector() {
  L = new long long[n + 1];
  S = new double[n + 1];
  SS = new double[n + 1];
  L[0] = 0;
  S[0] = 0;
  SS[0] = 0;
  for (unsigned long i = 1; i <= n; i++) {
    L[i] = L[i - 1] + (dat[i].te - dat[i].ts + 1);
    S[i] = S[i - 1] + (dat[i].te - dat[i].ts + 1) * dat[i].value;
    SS[i] = SS[i - 1]
        + (dat[i].te - dat[i].ts + 1) * dat[i].value * dat[i].value;
  }
}

/**
 * initialize the Gap- G-Vector
 */
void initGvector() {
  std::vector<unsigned long> gapVect;
  gapVect.push_back(0);
  for (unsigned long i = 1; i < n; i++) {
    if ((dat[i].group).compare(dat[i + 1].group) != 0
        || (dat[i].te + 1) != dat[i + 1].ts) {
      gapVect.push_back(i);
    }
  }
  //transform vector to array
  G = new unsigned long[gapVect.size()];
  std::copy(gapVect.begin(), gapVect.end(), G);
  numGaps = gapVect.size();

  gapVect.clear();
  gapVect.shrink_to_fit();
}

/**
 * get position of the last gap in the gap-vector
 * find the position of the right-most non-adjacent tuple pair
 * @param i tuple position for which the gap position is computed
 * @return position of the gap
 */
unsigned long getMaxGap(unsigned long i) {
  unsigned long lo = 0;
  unsigned long hi = numGaps;
  while (lo != hi) {
    unsigned long mid = lo + (hi - lo) / 2;
    unsigned long midval = G[mid];
    if (midval < i) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo == 0)
    return (0);
  if (lo > numGaps) {
    printf("error lo: %li, numgaps: %li", lo, numGaps);
    exit(-1);
  }
  return (G[lo - 1]);
}

/**
 * Compute the SSE of the Merge of the tuples between i and j into 1
 * @param i - start # of tuple
 * @param j - end # of tuple
 * @return error in merging tuples between i and j
 */
double computeSSE(long i, long j) {
  double s1, s2;
  s2 = SS[j] - SS[i - 1];
  s1 = S[j] - S[i - 1];
  return (s2 - s1 * s1 / (L[j] - L[i - 1]));
}

void printSplitPointGraph() {
  for (unsigned long i = 0; i < (n - c + 1); i++) {
    printf("node num = %li:", i);
    spgnode* startnode = splitCur[i];
    printf("[%p, %u,%u, %p]", startnode, startnode->indexnumber,
        startnode->numchilds, startnode->pathparent);
    while (startnode->pathparent != nullptr) {
      startnode = startnode->pathparent;
      printf("[%p, %u,%u, %p]", startnode, startnode->indexnumber,
          startnode->numchilds, startnode->pathparent);
    }
    printf("\n");
  }
}

/**
 * Generate test data
 */
double* generatedata(unsigned long n) {
  double* dataset = new double[n + 1];
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(5.0, 2.0);
  for (unsigned long i = 0; i < (n + 1); i++) {
    dataset[i] = rand() % 10000;
  }
  return (dataset);
}

/*
 * print the dataset
 */
void printdata() {
  printf("Dataset: \n");
  for (unsigned long i = 1; i <= n; i++)
    printf("G:%s val:%f [%li,%li]", dat[i].group.c_str(), dat[i].value,
        dat[i].ts, dat[i].te);
  printf("\n");
}

/**
 * Initialize error and splitpoint matrix
 */
void initEJMatrix() {
  errCurr = new double[n + 1]; //error matrix element 0 not used, indexed with the real tuple numbers
  errPrev = new double[n + 1];
  //initialize row with double infinity
  for (unsigned long j = 0; j < (n + 1); j++) {
    errCurr[j] = MAX;
    errPrev[j] = MAX;
  }

  splitNodePool = new std::stack<spgnode*>;
  //create an empty spgnode array for last two levels if diagonal pruning less nodes are needed
  splitCur = new spgnode*[n + 1];
  splitPrev = new spgnode*[n + 1];

  //initialize vector previous (level 1)
  for (unsigned long i = 1; i <= (n); i++) {
    spgnode* newnode = getNodeFromPool();
    newnode->indexnumber = i;
    newnode->numchilds = 0;
    newnode->pathparent = 0;
    splitPrev[i] = newnode;
#ifdef STATISTICS
    numnodes++;
#endif
  }
#ifdef STATISTICS
  if (numnodes > maxnodes) maxnodes = numnodes;
#endif

  //initialize current split array (for level k=2)
  for (unsigned long i = 1; i <= (n); i++) {
    spgnode* newnode = getNodeFromPool();
    newnode->indexnumber = i;
    newnode->numchilds = 0;
    newnode->pathparent = 0;
    splitCur[i] = newnode;
#ifdef STATISTICS
    numnodes++;
#endif
  }
#ifdef STATISTICS
  if (numnodes > maxnodes ) maxnodes = numnodes;
#endif
}

void pathpruning() {
  for (unsigned long ivar = 1; ivar <= (n); ivar++) {
    spgnode* startnode = splitPrev[ivar];
    while (startnode != nullptr && startnode->numchilds == 0) {
      spgnode* nextnode = startnode->pathparent;
      if (nextnode != nullptr) {
        nextnode->numchilds--;
        startnode->pathparent = nullptr;
      }
      deleteNodeToPool(startnode);
#ifdef STATISTICS
      numnodes--;
#endif
      startnode = nullptr;
      startnode = nextnode;
    }
  }
}

void insertSPGraph(long sourceindex, long destindex, long level) {
  std::unique_lock<std::mutex> graphlk(graphMutex);
  splitCur[sourceindex - level]->pathparent = splitPrev[destindex - level + 1];
  splitPrev[destindex - level + 1]->numchilds++;
  graphlk.unlock();
}

/*
 * searchlowerbound: performs a binary search in sseCompute returning the index of the leftmost element that is lower than the searchvalue
 * note: sseCompute is in decreasing order!
 * start: start index of search space
 * stop: end index of search space
 * searchvalue: value to search
 * column: reference Column for sseComputation, i.e. the index of the cell for which we compute the errors
 */
unsigned long searchlowerboundjaga(unsigned long start, unsigned long stop,
    double searchvalue, unsigned long column) {
  //searches lower bound like in jagadish paper!
  unsigned long lo = start;
  unsigned long hi = stop;
  unsigned long mid;
  double midval;
  if (computeSSE(start + 1, column) <= searchvalue)
    return (start);
  while (lo != hi) {
    mid = lo + std::ceil((hi - lo) / 2.0); //round up since we search in descending array

    /* the sse computation for index i starts at i+1 and ends at column
     our reference for index is always the index of the E value, not SSE
     */
    midval = computeSSE(mid + 1, column);
    if (midval <= searchvalue) {
      hi = mid - 1;
    } else {
      lo = mid;
    }
  }
  return (lo);
}

/*
 * searchupperbound
 * start: start index of search space
 * stop: end index of search space
 * searchvalue: value to search
 */
unsigned long searchupperboundjaga(unsigned long start, unsigned long stop,
    double searchvalue) {
  unsigned long lo = start;
  unsigned long hi = stop;
  while (lo != hi) {
    unsigned long mid = int(lo + (hi - lo) / 2.0);
    double midval = errPrev[mid];
    if (midval <= searchvalue) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return (hi);
}

void threadComputeElement(int threadnum) {
  double sserr;
  while (!tFinished) {
    std::unique_lock<std::mutex> wqlk(multiMutex);
    while ((jcurr.load() > lineJmax && tWorkersBusy > 0) || !tStartrow) // used to avoid spurious wakeups
    {
      // wait until all workers for the current row have t_finished
      // controller will us wake up
      workerStart.wait(wqlk);
    }

    if (jcurr.load() <= lineJmax && tStartrow && !tFinished) {
      ++tWorkersBusy; // increment counter of busy
      int blocksize = n / numthreads;
      unsigned long jstart = jcurr.fetch_add(blocksize);
      //release jcurr for other processes
      wqlk.unlock();

      unsigned long endj = jstart + blocksize - 1;
      if (endj > lineJmax)
        endj = lineJmax;

      for (unsigned long j = jstart; j <= endj; j++) {
        //now do the computation
        errCurr[j] = MAX;
        unsigned long xmax = j - 1;
        unsigned long xmin = (i - 1 < getMaxGap(j) ? getMaxGap(j) : i - 1);
        unsigned long optsplit = xmax;
        if (numGaps > (i - 1) && (G[i - 1] == xmin)) {
          //split has to be at jmin, insert it directly without loop
          optsplit = xmin;
          errCurr[j] = errPrev[xmin] + computeSSE(xmin + 1, j);
        } else {
          optsplit = xmax;
          unsigned long seedpos = xmin;
          double seederr = MAX;
          //our seed search
          //reference in splitpref -i not -(i+1) since we access previous vector
          if (splitPrev[j - 1] != nullptr
              && splitPrev[j - 1]->pathparent != nullptr) {
            unsigned long testpos = splitPrev[j - 1]->pathparent->indexnumber;
            double errtestpos = errPrev[testpos] + computeSSE(testpos + 1, j);
            if (errtestpos < seederr) {
              seederr = errtestpos;
              seedpos = testpos;
            }
          }
          if (j > i && splitPrev[j - 2] != nullptr
              && splitPrev[j - 2]->pathparent != nullptr) { //if i ==j element does not exist! (only the case if c = n)
            unsigned long testpos = splitPrev[j - 2]->pathparent->indexnumber;
            double errtestpos = errPrev[testpos] + computeSSE(testpos + 1, j);
            if (errtestpos < seederr) {
              seederr = errtestpos;
              seedpos = testpos;
            }
          }
          //now do the iterative search
          if (seedpos < xmin)
            seedpos = xmin;
          seederr = errPrev[seedpos] + computeSSE(seedpos + 1, j);
          if (seederr != 0) {
            unsigned long xm = xmax;
            xm = searchlowerboundjaga(xmin, xmax, seederr, j);
            double seed = seederr - errPrev[xm];
            unsigned long xmprev = xmax;
            while (xm != xmprev) {
              seed = seederr - errPrev[xm];
              xmprev = xm;
              xm = searchlowerboundjaga(xmprev, xmax, seed, j);
            }
            xmin = xm;
          }

          //in case of binary search loop upwards
          for (unsigned long x = xmin; x <= xmax; x++) {
            sserr = computeSSE(x + 1, j);
            if (errPrev[x] + sserr < errCurr[j]) {
              errCurr[j] = errPrev[x] + sserr;
              optsplit = x;
            }
            //we loop upwards, break condition on errPrev
            if (errPrev[x] > errCurr[j])
              break;
          }
        }

        splitCur[j]->pathparent = splitPrev[optsplit];
        (splitPrev[optsplit]->numchilds)++;
      } //loop if blocks of more than 1 element are computed

      tWorkersBusy--;
    } else if (tFinished) {
      wqlk.unlock();
      break;
    } else
      wqlk.unlock();
  }
  tWorkersCount--;
}

/**
 * Compute the Error and Split point matrix
 */
void computeMatrices() {
  for (unsigned long l = 1; l <= (n - c + 1); l++) {
    errPrev[l] = computeSSE(1, l);
  }
  tFinished = false; //true then all workers will exit
  tWorkersBusy = 0;
  tStartrow = false; //true then all workers will start the row computation

  for (int twc = 0; twc < numthreads; twc++) {
    tWorkersCount++;
    tWorkers.push_back(std::thread(threadComputeElement, twc));
  }

  for (i = 2; i <= c; i++) {
    // thread initialization
    tStartrow = false; //to prevent that threads are starting too early (before vars are initialized)
    lineJmax = n;
    unsigned long imaxGapVector = n;
    if (i < numGaps) {
      imaxGapVector = G[i];
    }
    lineJmax = std::min((n - (c - i)), imaxGapVector);

    //initialize jcurr
    jcurr = i;
    errCurr[i] = 0;
    if (i == c)
      jcurr = n; //last row pruning
    //start the workers
    {
      std::unique_lock<std::mutex> wqlk(multiMutex);
      tStartrow = true;
      workerStart.notify_all();
      wqlk.unlock();
    }

    //Wait until workers have finished
    while (jcurr <= lineJmax || tWorkersBusy > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      //workerHasFinished.wait(wqlk);
    }

    //ensure all workers are in waiting status
    tStartrow = false;

    if (i < c) { //switch error Array and initialize
      double *tmperr = errCurr;
      errCurr = errPrev;
      errPrev = tmperr;
      // in case of graph with binary search we need initialization of errCurr otherwise in the next we search in 0 values
      for (unsigned long j = 1; j <= (n); j++) {
        errCurr[j] = MAX;
      }

      //pathpruning but only if after level 2 and before last level
      if (i >= 2) {
#ifdef DEBUG
        printf("dopathpruning %li\n", i);
#endif
        pathpruning();
      }
      //initialize next graph array
      spgnode** tmpsplit = splitPrev;
      splitPrev = splitCur;        // last becomes prev
      splitCur = tmpsplit;
      //i+1 since we add for next level

      for (unsigned long j = 1; j <= (n); j++) {
        spgnode* newnode = getNodeFromPool();
        newnode->indexnumber = j;
        newnode->numchilds = 0;
        newnode->pathparent = nullptr;
        splitCur[j] = newnode;
#ifdef STATISTICS
        numnodes++;
#endif
      }
#ifdef STATISTICS
      if (numnodes > maxnodes) maxnodes = numnodes;
#endif
    }
  }

  //notify all waiting threads to finish and exit while(true) loop
  tFinished = true;
  tStartrow = true;
  while (tWorkersCount > 0) {
    workerStart.notify_all();
  }

  for (auto& t : tWorkers) {
    t.join();
  }
  tWorkers.clear();
}

/*
 * Returns the split path
 */
std::string getSplitPath() {
  spgnode* startnode = splitCur[n];
  string splitpath = "";
  while (startnode->pathparent != nullptr) {
    startnode = startnode->pathparent;
    splitpath += std::to_string(startnode->indexnumber) + ",";
  }
  splitpath += "0,\n";
  return (splitpath);
}

/**
 * Display message for usage
 *
 */
void display_usage(void) {
  fprintf(stderr, "OKS - optimum k segments multi threading implementation\n");
  fprintf(stderr,
      "usage: -i inputfile -n inputsize -c reductionsize [-o] [-p]\n");
  fprintf(stderr, " -o show resulting split path\n");
  fprintf(stderr, " -p show data\n");
  exit(EXIT_FAILURE);
}

/*
 * Main
 */
int main(int argc, char** argv) {
  //save arguments for output
  std::string cmdline;
  for (int i = 0; i < argc; i++) {
    cmdline.append(argv[i]);
    cmdline.append(" ");
  }
  //parse arguments
  int opt = 0;
  std::string csvFileName;
  int iflag = 0;
  int nflag = 0;
  int cflag = 0;
  int oflag = 0;
  int tflag = 0;
  int pflag = 0;
  while ((opt = getopt(argc, argv, "i:n:c:t:oph?")) != -1) {
    switch (opt) {
    case 'i':
      iflag++;
      csvFileName.assign(optarg);
      break;
    case 'n':
      nflag++;
      n = atoi(optarg);
      break;
    case 'c':
      cflag++;
      c = atoi(optarg);
      break;
    case 't':
      tflag++;
      numthreads = atoi(optarg);
      break;
    case 'o':
      oflag++;
      break;
    case 'p':
      pflag++;
      break;
    case 'h': /* fall-through is intentional */
    case '?':
      display_usage();
      break;
    default:
      /* You won't actually get here. */
      break;
    }
  }
  static char usage[] =
      "usage: %s -i dataset -n inputsize -c reductionsize -t numberofthreads [-o] [-p]\n";
  if (iflag == 0 && cflag == 0 && nflag == 0) {
    //no flags set, use defaults
    fprintf(stderr, "no size parameters set, using default values\n");
    fprintf(stderr, usage, argv[0]);
    n = 1000;
    nflag = 1;
    c = 10;
    cflag = 1;
    tflag = 1;
    numthreads = 2;
    oflag = 1;
    pflag = 0;
    iflag = 1;
    csvFileName = "./dataset/webkit.csv";
    printf("Options: i:%s n:%li c:%li \n", csvFileName.c_str(), n, c);
  } else {
    if (iflag == 0) {
      /* -i is mandatory */
      fprintf(stderr, "%s: missing -i option\n", argv[0]);
      fprintf(stderr, usage, argv[0]);
      exit(1);
    }
    if (nflag == 0) {
      /* -n is mandatory */
      fprintf(stderr, "%s: missing -n option\n", argv[0]);
      fprintf(stderr, usage, argv[0]);
      exit(1);
    }
    if (cflag == 0) {
      /* -c is mandatory */
      fprintf(stderr, "%s: missing -c option\n", argv[0]);
      fprintf(stderr, usage, argv[0]);
      exit(1);
    }
    if (tflag == 0) {
      /* -t is not mandatory - use standard value */
      printf("%s: missing -t option\n", argv[0]);
      printf("using standard value: t=2");
      numthreads = 2;
    }
  }
  if (iflag == 1) {
    //load from database
    dat = readCSV(csvFileName, n);
  }

  if (pflag > 0)
    printdata();

  initEJMatrix();
  initLSSSvector();
  initGvector();
  if (c < numGaps) {
    printf("\n\nReduction size too small; min size %lu\n\n", numGaps);
    exit(-1);
  }
  auto begin = std::chrono::high_resolution_clock::now();
  computeMatrices();
  auto end = std::chrono::high_resolution_clock::now();

  if (oflag > 0) {
    string splitpath = getSplitPath();
    printf("Splitpath: %s\n", splitpath.c_str());
  }
  printf("\"%s\", %s, %s, %ld, %ld, %lld", cmdline.c_str(), argv[0],
      csvFileName.c_str(), n, c,
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
  //cleanup - remove all remaining spgnodes from graph
#ifdef STATISTICS
  printf(", %i\n",maxnodes);
#endif

  pathpruning();
  //remove nodes along split path from graph
  spgnode* startnode = splitCur[n];
  while (startnode != NULL) {
    spgnode* nextnode = startnode->pathparent;
    delete startnode;
    startnode = nextnode;
  }

  return (0);
}
