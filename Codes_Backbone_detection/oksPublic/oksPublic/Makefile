CC=g++
INCLUDES        = 
CFLAGS= -Wformat=0 -Wall ${INCLUDES} -std=c++0x
LIBS            = 
LDLIBS          = 
OBJECTFILES = data.o
DEST = ./binrt/

all: rt statistics debug
rt: CFLAGS += -O3 -march=native
rt: DEST = ./binrt/
statistics: CFLAGS += -O3 -DSTATISTICS
statistics: DEST = ./binstat/
debug: CFLAGS += -DDEBUG -ggdb -O0 -ftrapv -pg
debug: DEST = ./debug/

#Targets

PTA =  $(DEST)ptamat $(DEST)ptamatnaive $(DEST)ptamatdp $(DEST)ptamatdpnogaps $(DEST)ptagraph  
PTA += $(DEST)ptamatjag $(DEST)ptamatjagplus $(DEST)ptamatour
PTA += $(DEST)ptagraphjag $(DEST)ptagraphjagplus $(DEST)ptagraphour
PTAMULTI = $(DEST)ptamultimat $(DEST)ptamultimatdp $(DEST)ptamultigraph 
PTAMULTI += $(DEST)ptamultimatjag $(DEST)ptamultimatjagplus $(DEST)ptamultimatour
PTAMULTI += $(DEST)ptamultigraphjag $(DEST)ptamultigraphjagplus $(DEST)ptamultigraphour

rt: data.o 
rt: $(PTA)
rt: $(PTAMULTI)
rt: oksm

statistics: data.o 
statistics: $(PTA)
statistics: $(PTAMULTI)
statistics: oksm

debug: data.o 
debug: $(PTA)
debug: $(PTAMULTI)
#debug: $(middlebucket)
debug: oksm

data.o: data.cpp data.h
	$(CC) $(CFLAGS) -c -o data.o data.cpp

### Optimum K Segments -multithread
oksm:  oksm.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -O3 -march=native  $(LIBS) -o $(DEST)oksm oksm.cpp $(OBJECTFILES) $(LDLIBS) -lpthread
	$(CC) $(CFLAGS) -DDEBUG -ggdb -O0 -ftrapv -pg $(LIBS) -o debug/oksm oksm.cpp $(OBJECTFILES) $(LDLIBS) -lpthread


########### PTA
$(DEST)ptamatnaive: pta.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK $(LIBS) -o $(DEST)ptamatnaive pta.cpp $(OBJECTFILES) $(LDLIBS)
$(DEST)ptamat: pta.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK -DGAPPRUNING $(LIBS) -o $(DEST)ptamat pta.cpp $(OBJECTFILES) $(LDLIBS)
$(DEST)ptamatdpnogaps: pta.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK -DDP $(LIBS) -o $(DEST)ptamatdpnogaps pta.cpp $(OBJECTFILES) $(LDLIBS)
$(DEST)ptamatdp: pta.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK -DDP -DGAPPRUNING $(LIBS) -o $(DEST)ptamatdp pta.cpp $(OBJECTFILES) $(LDLIBS)
$(DEST)ptagraph: pta.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK -DDP -DGAPPRUNING -DGRAPH -DNODEPOOL $(LIBS) -o $(DEST)ptagraph pta.cpp $(OBJECTFILES) $(LDLIBS)
$(DEST)ptamatjag: pta.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK -DBINSEARCH -DSEEDJAG $(LIBS) -o $(DEST)ptamatjag pta.cpp $(OBJECTFILES) $(LDLIBS)
$(DEST)ptamatjagplus: pta.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK -DBINSEARCH -DSEEDOPTPREV $(LIBS) -o $(DEST)ptamatjagplus pta.cpp $(OBJECTFILES) $(LDLIBS)
$(DEST)ptamatour: pta.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK -DDP -DGAPPRUNING -DBINSEARCH -DSEEDOUR $(LIBS) -o $(DEST)ptamatour pta.cpp $(OBJECTFILES) $(LDLIBS)
$(DEST)ptagraphjag: pta.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK  -DGRAPH -DNODEPOOL -DBINSEARCH -DSEEDJAG $(LIBS) -o $(DEST)ptagraphjag pta.cpp $(OBJECTFILES) $(LDLIBS)
$(DEST)ptagraphjagplus: pta.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK  -DGRAPH -DNODEPOOL -DBINSEARCH -DSEEDOPTPREV $(LIBS) -o $(DEST)ptagraphjagplus pta.cpp $(OBJECTFILES) $(LDLIBS)
$(DEST)ptagraphour: pta.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK -DDP -DGAPPRUNING -DGRAPH -DNODEPOOL -DBINSEARCH -DSEEDOUR $(LIBS) -o $(DEST)ptagraphour pta.cpp $(OBJECTFILES) $(LDLIBS)

############ PTA MULTI
$(DEST)ptamultimat: ptamulti.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK -DGAPPRUNING $(LIBS) -o $(DEST)ptamultimat ptamulti.cpp $(OBJECTFILES) $(LDLIBS) -lpthread
$(DEST)ptamultimatdp: ptamulti.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK -DDP -DGAPPRUNING $(LIBS) -o $(DEST)ptamultimatdp ptamulti.cpp $(OBJECTFILES) -lpthread
$(DEST)ptamultigraph: ptamulti.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK -DDP -DGAPPRUNING -DGRAPH -DNODEPOOL $(LIBS) -o $(DEST)ptamultigraph ptamulti.cpp $(OBJECTFILES) -lpthread
$(DEST)ptamultimatjag: ptamulti.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK -DGAPPRUNING -DBINSEARCH -DSEEDJAG $(LIBS) -o $(DEST)ptamultimatjag ptamulti.cpp $(OBJECTFILES) -lpthread
$(DEST)ptamultimatjagplus: ptamulti.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK -DGAPPRUNING -DBINSEARCH -DSEEDOPTPREV $(LIBS) -o $(DEST)ptamultimatjagplus ptamulti.cpp $(OBJECTFILES) -lpthread
$(DEST)ptamultimatour: ptamulti.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK -DDP -DGAPPRUNING -DBINSEARCH -DSEEDOUR $(LIBS) -o $(DEST)ptamultimatour ptamulti.cpp $(OBJECTFILES) -lpthread
$(DEST)ptamultigraphjag: ptamulti.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK -DGAPPRUNING -DGRAPH -DDNODEPOOL -DBINSEARCH -DSEEDJAG $(LIBS) -o $(DEST)ptamultigraphjag ptamulti.cpp $(OBJECTFILES) -lpthread
$(DEST)ptamultigraphjagplus: ptamulti.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK -DGAPPRUNING -DGRAPH -DDNODEPOOL -DBINSEARCH -DSEEDOPTPREV $(LIBS) -o $(DEST)ptamultigraphjagplus ptamulti.cpp $(OBJECTFILES) -lpthread
$(DEST)ptamultigraphour: ptamulti.cpp $(OBJECTFILES)
	$(CC) $(CFLAGS) -DEARLYBREAK -DDP -DGAPPRUNING  -DGRAPH -DDNODEPOOL -DBINSEARCH -DSEEDOUR $(LIBS) -o $(DEST)ptamultigraphour ptamulti.cpp $(OBJECTFILES) -lpthread

clean: 	
	rm -rf data.o
	rm -rf ./binrt/*
	rm -rf ./binstat/*
	rm -rf ./debug/*
