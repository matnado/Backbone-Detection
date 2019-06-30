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
 * File:   pta.cpp
 * Author: Giovanni Mahlknecht Free University of Bozen/Bolzano
 *         giovanni.mahlknecht@inf.unibz.it
 * 
 * Created on November 19, 2015, 3:09 PM
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <random>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <utility>
#include "data.h"

//#define STATISTICS // to visualize statistics summary
//#define EARLYBREAK // to enable or disable early break of inner j loop
//#define GRAPH // to enable split point graph, otherwise J matrix is used
//#define NODEPOOL // to use a nodepool instead of single creation and deletion of nodes
using namespace std;

const double MAX = std::numeric_limits<double>::max();

itatuple* dat; //input data
unsigned long n; //dataset size
unsigned long c; //size to reduce to
#ifdef GRAPH
struct spgnode {
  long indexnumber;
  spgnode* pathparent;
  long numchilds;
};
//structure for nodes in the graph
#ifdef NODEPOOL
std::stack<spgnode*> *splitNodePool; //pool for storing unused graph nodes
#endif
spgnode **splitCur; //references to current split nodes (the level)
spgnode **splitPrev;
#else
long** J; //split point matrix
#endif
double* errCurr; //last two rows of error matrix
double* errPrev;
double* S; //linear sum
double* SS; //squared sum
long long* L; //L vector for weighting error tuple duration
unsigned long * G; //Gap vector
unsigned long numGaps;
#ifdef STATISTICS
unsigned long long ssecount;
unsigned long long innerloopcount;
unsigned long long maxnodes; //maximum number of nodes in graph
unsigned long long numnodes;//number of nodes in graph
unsigned long long nodescreated;// total number of nodes created (only if nodepool is used)
unsigned long long binsearchsteps;//number of binary search steps
unsigned long long seedsteps;//number of seed search steps
#endif

#ifdef NODEPOOL
/**
 * retrieve a new spgnode from the pool
 * @return a pointer to a spgnode
 */
spgnode* getNodeFromPool() {
  spgnode* newnode;
  if (splitNodePool->empty()) {
    /* create new node and return pointer to it */
    newnode = new spgnode;
#ifdef STATISTICS
    nodescreated++;
#endif
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
#endif

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
      //printf("%li",i);
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
#ifdef STATISTICS
  ssecount++;
#endif
  double s1, s2;
  s2 = SS[j] - SS[i - 1];
  s1 = S[j] - S[i - 1];
  return (s2 - s1 * s1 / (L[j] - L[i - 1]));
}

#ifdef GRAPH
void printSplitPointGraph() {
  for (unsigned long i = 0; i < (n - c + 1); i++) {
    printf("node num = %li:", i);
    spgnode* startnode = splitCur[i];
    printf("[%p, %li,%li, %p]", startnode, startnode->indexnumber,
        startnode->numchilds, startnode->pathparent);
    while (startnode->pathparent != NULL) {
      startnode = startnode->pathparent;
      printf("[%p, %li,%li, %p]", startnode, startnode->indexnumber,
          startnode->numchilds, startnode->pathparent);
    }
    printf("\n");
  }
}
#endif

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

#ifndef GRAPH
/*
 * Print splitpoint matrix
 */
void printJ() {
  printf("Split Point Matrix J\n");
  for (unsigned long i = 0; i <= c; i++) {
    for (unsigned long j = 0; j <= n; j++) {
      printf("%li ", J[i][j]);
    }
    printf("\n");
  }
}
#endif

/**
 * Initialize error and splitpoint matrix
 */
void initEJMatrix() {
  errCurr = new double[n + 1];
  errPrev = new double[n + 1];
  //initialize row with double infinity
  for (unsigned long j = 0; j < (n + 1); j++) {
    errCurr[j] = MAX;
    errPrev[j] = MAX;
  }

#ifdef GRAPH
#ifdef NODEPOOL
  splitNodePool = new std::stack<spgnode*>;
#endif
  //create an empty spgnode array for last two levels if diagonal pruning less nodes are needed
#ifdef DP
  splitCur = new spgnode *[n - c + 1];
  splitPrev = new spgnode *[n - c + 1];
#else
  splitCur = new spgnode *[n];
  splitPrev = new spgnode *[n];
#endif

  //initialize vector previous (level 1)
#ifdef DP
  for (unsigned long i = 0; i < (n - c + 1); i++) {
#else //noDP
    for (unsigned long i = 0; i < (n); i++) {
#endif
#ifdef NODEPOOL
      spgnode* newnode = getNodeFromPool();
#else
      spgnode* newnode = new spgnode;
#endif
      newnode->indexnumber = i + 1;
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
#ifdef DP
    for (unsigned long i = 0; i < (n - c + 1); i++) {
#else //noDP
      for (unsigned long i = 0; i < (n); i++) {
#endif
#ifdef NODEPOOL
        spgnode* newnode = getNodeFromPool();
#else
        spgnode* newnode = new spgnode;
#endif
        newnode->indexnumber = i + 2;
        newnode->numchilds = 0;
        newnode->pathparent = 0;
        splitCur[i] = newnode;
#ifdef STATISTICS
        numnodes++;
#endif
      }
#ifdef STATISTICS
      if (numnodes > maxnodes) maxnodes = numnodes;
#endif
#else
  J = new long*[c + 1];
  for (unsigned long i = 0; i < c + 1; i++) {
    J[i] = new long[n + 1];
    //initialize each row with 0
    for (unsigned long j = 0; j < (n + 1); j++) {
      J[i][j] = 0;
    }
  }
#endif
}

#ifdef GRAPH
void pathpruning() {
  for (unsigned long i = 0; i < (n - c + 1); i++) {
    spgnode* startnode = splitPrev[i];
    while (startnode != NULL && startnode->numchilds == 0) {
      spgnode* nextnode = startnode->pathparent;
      if (nextnode != NULL) {
        nextnode->numchilds--;
        startnode->pathparent = NULL;
      }
#ifdef NODEPOOL
      deleteNodeToPool(startnode);
#else
      delete startnode;
#endif
      startnode = NULL;
#ifdef STATISTICS
      numnodes--;
#endif
      startnode = nextnode;
    }
  }
}

void insertSPGraph(long sourceindex, long destindex, long level) {
  splitCur[sourceindex - level]->pathparent =
  splitPrev[destindex - level + 1];
  splitPrev[destindex - level + 1]->numchilds++;
}

#endif

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
#ifdef STATISTICS
    binsearchsteps++;
#endif
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
#ifdef STATISTICS
    binsearchsteps++;
#endif
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

/**
 * Compute the Error and Split point matrix
 */
void computeMatrices() {
  double sserr;
  unsigned long optsplit;
  unsigned long j;
  unsigned long i;
#ifdef DP
  for (j = 1; j <= (n - (c - 1)); j++) {
#else
  for (j = 1; j <= n; j++) {
#endif
    errPrev[j] = computeSSE(1, j);
  }
  for (i = 2; i <= c; i++) {
    unsigned long jmax = n;
#ifndef DP
#ifdef GAPPRUNING
    if (i < numGaps) jmax = G[i];
#else
    jmax = n;
#endif
#else
    //dp - use gap vector and diagonal pruning, take the lower value
#ifdef GAPPRUNING
    unsigned long imaxGapVector = n;
    if (i < numGaps) {
      imaxGapVector = G[i]; //-1 since vector stats at 0 to
    }
    jmax = std::min((n - (c - i)), imaxGapVector);
#else
    jmax=n-(c-i);
#endif
#endif
    for (j = i; j <= jmax; j++) {
#ifdef DP
      if (i == c)
      j = n; //last row pruning
#endif
      errCurr[j] = MAX;
      unsigned long xmax = j - 1;
#ifdef GAPPRUNING
      unsigned long xmin = (i - 1 < getMaxGap(j) ? getMaxGap(j) : i - 1);
#else
      unsigned long xmin = i - 1;
#endif
      optsplit = xmax;
      if (numGaps > (i - 1) && (G[i - 1] == xmin)) {
        //split has to be at jmin, insert it directly without loop
        optsplit = xmin;
        errCurr[j] = errPrev[xmin] + computeSSE(xmin + 1, j);
      } else {
        optsplit = xmax;
#ifdef BINSEARCH
        unsigned long seedpos = n;
        double seederr;
#ifdef GRAPH
#ifdef SEEDOPTPREV
        if (i == 2) {
          // we have no edge to nodes for k=1 they are superfluous
          seedpos = 2;
        } else {
          //-(k-1) since the previous level is shifted by k-1 positions i-1 because we want the node above left, not the node above directly (diagonal pruned)
          //since we have gaps it is possible that we have no optimal point for i-1 in the previous level,
          //we are in the first row of the computation of the group bucket, therefore optprevpos does not exist, set it to jmin
          if (splitPrev[j-1 - (i-1)]->pathparent!=nullptr) {
            seedpos =
            splitPrev[j - 1 - (i-1)]->pathparent->indexnumber;
          }
          else seedpos = xmin;
        }
#endif //seedoptprev
#ifdef SEEDJAG
        //loop n to n-(n/k) downwards and take minimum
        //attention seedjag not compatible with diagonal pruning!
        seedpos = n-1;
        seederr = MAX;
        for(unsigned long seedloop = j-1; seedloop >= (n/c >= j-i ? i-1 : j-n/c);seedloop--) {
#ifdef STATISTICS
          seedsteps++;
#endif

          double seedval = errPrev[seedloop]+computeSSE(seedloop+1,n);
          if (seedval < seederr) {
            seederr = seedval;
            seedpos = seedloop;
          }
        }
#endif //SEEDJAG
#ifdef SEEDOUR
        //upper row j-delta ... j
        unsigned int delta=1;
        long signedleft =j - delta - (i - 1);//left and right are not cell values but indices in splitPrev!
        unsigned long left = (signedleft < 0 ? 0 : signedleft);
        long signedright = j - (i - 1);
        unsigned long right = (signedright < 0 ? 0 : signedright);
        if (right>(n -(c))) right=(n-(c));//only with diagonal pruning
        if (left<0) left=0;

        double sumerr = MAX;
        for (unsigned int loop = left; loop <= right; loop++)
        {
#ifdef STATISTICS
          seedsteps++;
#endif
          //test if splitprev[loop] has next pointer exists
          if (splitPrev[loop]->pathparent != nullptr) {
            unsigned long testpos = splitPrev[loop]->pathparent->indexnumber;
            double errtestpos = errPrev[testpos]
            + computeSSE(testpos + 1, j);
            if (errtestpos < sumerr) {
              sumerr=errtestpos;
              seedpos = testpos;
            }
          }
        }

#endif //SEEDOUR
#else //MATRIX
#ifdef SEEDOPTPREV
        seedpos = J[i - 1][j - 1];
#endif
#ifdef SEEDJAG
        //loop n to n-(n/k) downwards and take minimum
        //attention seedjag not compatible with diagonal pruning!
        seedpos = n-1;
        seederr = MAX;
        for(unsigned long seedloop = j-1; seedloop >= (n/c >= j-i ? i-1 : j-n/c);seedloop--) {
#ifdef STATISTICS
          seedsteps++;
#endif

          double seedval = errPrev[seedloop]+computeSSE(seedloop+1,n);
          if (seedval < seederr) {
            seederr = seedval;
            seedpos = seedloop;
          }
        }
#endif //SEEDJAG
#ifdef SEEDOUR
        //upper row j-delta ... j
        unsigned int delta=1;
        long signedleft =j - delta;//left and right are not cell values but indices in splitPrev!
        unsigned long left = (signedleft < 0 ? 0 : signedleft);
        int signedright = j;
        unsigned long right = (signedright < 0 ? 0 : signedright);
        if (right>(n -(c-i))) right=(n-(c-i));//only with diagonal pruning
        if (left<i) left=i;

        double sumerr = MAX;
        for (unsigned int loop = left; loop <= right; loop++)
        {
#ifdef STATISTICS
          seedsteps++;
#endif
          //test if splitprev[loop] has next pointer exists
          unsigned long testpos = J[i-1][loop];
          double errtestpos = errPrev[testpos]
          + computeSSE(testpos + 1, j);
          if (errtestpos < sumerr) {
            sumerr=errtestpos;
            seedpos = testpos;
          }
        }

#endif //SEEDOUR
#endif //matrix implementation
        if (seedpos < xmin)
        seedpos = xmin;
        seederr = errPrev[seedpos] + computeSSE(seedpos + 1, j);
        if (seederr != 0) {
          unsigned long xm = xmax;
          xm = searchlowerboundjaga(xmin, xmax, seederr, j);
          double seed = seederr - errPrev[xm];
          unsigned long xmprev = xmax;
          while (xm != xmprev) {
#ifdef STATISTICS
            binsearchsteps++;
#endif
            seed = seederr - errPrev[xm];
            xmprev = xm;
            xm = searchlowerboundjaga(xmprev, xmax, seed, j);
          }
          xmin = xm;
        }
        //in case of binary search loop upwards, else loop downwards
        for (unsigned long x = xmin; x <= xmax; x++) {
#else
        for (unsigned long x = (xmax); x >= xmin; x--) {
#endif //binary search

#ifdef STATISTICS
          innerloopcount++;
#endif
          sserr = computeSSE(x + 1, j);
          if (errPrev[x] + sserr < errCurr[j]) {
            errCurr[j] = errPrev[x] + sserr;
            optsplit = x;
          }
#ifdef EARLYBREAK
#ifdef BINSEARCH
          //we loop upwards, break condition on errPrev
          if (errPrev[x] > errCurr[j])
          break;
#else
          //loop down, break condition sserr
          if (sserr > errCurr[j]) break;
#endif
#endif
        }
      }
#ifdef GRAPH
      insertSPGraph(j, optsplit, i);
#else
      J[i][j] = optsplit;
#endif
    }
    //switch error Array and initialize
    double *tmp = errCurr;
    errCurr = errPrev;
    errPrev = tmp;

#ifdef GRAPH
    //pathpruning but only if after level 2 and before last level
    if (i >= 2 && i < c)
    pathpruning();

    //initialize next graph array
    if (i < c) {
      spgnode** tmp = splitPrev;
      splitPrev = splitCur;        // last becomes prev
      splitCur = tmp;
#ifdef DP
      for (unsigned long j = 0; j < (n - c + 1); j++) {
#else //noDP
        for (unsigned long j = 0; j < (n - i + 1); j++) {
#endif
#ifdef NODEPOOL
          spgnode* newnode = getNodeFromPool();
#else
          spgnode* newnode = new spgnode;
#endif
          newnode->indexnumber = j + i + 1; //+1 since we add for the next level (i+1)
          newnode->numchilds = 0;
          newnode->pathparent = NULL;
          splitCur[j] = newnode;
#ifdef STATISTICS
          numnodes++;
#endif
        }
#ifdef STATISTICS
        if (numnodes > maxnodes)
        maxnodes = numnodes;
#endif
      }
#endif
  }
}

/*
 * Returns the split path
 */
std::string getSplitPath() {
#ifdef GRAPH
  spgnode* startnode = splitCur[n - c];
  string splitpath = "";
  while (startnode->pathparent != NULL) {
    startnode = startnode->pathparent;
    splitpath += std::to_string(startnode->indexnumber) + ",";
  }
  splitpath += "0,\n";
#else
  long ci = c;
  long splitPoint;
  string splitpath = "";
  long endcounter = n;
  while (ci > 0) {
    splitPoint = J[ci][endcounter];
    splitpath += std::to_string(splitPoint) + ",";
    endcounter = splitPoint;
    ci--;
  }
  splitpath += "\n";
#endif
  return (splitpath);
}

/**
 * Display message for usage
 *
 */
void display_usage(void) {
  fprintf(stderr, "pta - computes parsimonious temporal aggregation\n");
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
  int pflag = 0;
  while ((opt = getopt(argc, argv, "i:n:c:f:oph?")) != -1) {
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
      "usage: %s -i dataset -n inputsize -c reductionsize [-o]\n";
  if (iflag == 0 && cflag == 0 && nflag == 0) {
    //no flags set, use defaults
    fprintf(stderr, "no size parameters set, using default values\n");
    fprintf(stderr, usage, argv[0]);
    n = 1000;
    nflag = 1;
    c = 10;
    cflag = 1;
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
  }
  if (iflag == 1) {
    //load from database
    dat = readCSV(csvFileName, n);
  }

  if (pflag > 0)
    printdata();
#ifdef STATISTICS
  /* initialize statistics counter */
  ssecount = 0;
  innerloopcount = 0;
  binsearchsteps = 0; //binary search steps
  numnodes = 0;//initialize graph size counter
  maxnodes = 0;
#endif

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
    printf("Error: %f\n", errCurr[n - 1]);
    printf("Splitpath: %s\n", splitpath.c_str());
  }
  printf("\"%s\", %s, %s, %ld, %ld, %lld , ", cmdline.c_str(), argv[0],
      csvFileName.c_str(), n, c,
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
#ifdef STATISTICS
  printf("%llu , ", ssecount);
  printf("%llu , ", innerloopcount);
#if defined(GRAPH)
  printf("%llu , ", maxnodes);
  printf("%llu , ", numnodes);
#else
  printf("0 , ");
  printf("0 , ");
#endif
#if defined(NODEPOOL)
  printf("%llu , ", nodescreated);
#else
  printf("0 , ");
#endif
#ifdef BINSEARCH
  printf("%llu , %llu", binsearchsteps, seedsteps);
#else
  printf("0 , 0");
#endif
#endif
  printf("\n");
  //cleanup - remove all remaining spgnodes from graph
#ifdef GRAPH
  pathpruning();

  //remove nodes along split path from graph
  spgnode* startnode = splitCur[n - c];
  while (startnode != NULL) {
    spgnode* nextnode = startnode->pathparent;
    delete startnode;
    startnode = nextnode;
  }

  //delete all nodes in splitLast; - don't remove last element since it is already removed
  for (unsigned long i = 0; i < (n - c); i++) {
    delete splitCur[i];
  }

  //free used arrays
  delete[] splitCur;
  delete[] splitPrev;
#ifdef NODEPOOL
  //free also node pool vector
  while (!splitNodePool->empty()) {
    spgnode* node = splitNodePool->top();
    delete node;
    splitNodePool->pop();
  }
  delete splitNodePool;
#endif
#else
  //free split point matrix J
  for (unsigned long i = 0; i <= c; i++) {
    delete[] J[i];
  }
  delete[] J;
#endif
  delete[] errCurr;
  delete[] errPrev;
  delete[] SS;
  delete[] S;
  delete[] L;
  delete[] dat;
  delete[] G;
  return (0);
}
