CC  = g++
SRC = *.cpp

EXE = training

ifeq ($(OS), Windows_NT)
	EXT = .exe
else
	EXT = 
endif

WFLAGS = -Wall
RFLAGS = $(WFLAGS) -std=c++17 -O3

ifeq ($(EXT), .exe)
	RFLAGS += -static -static-libgcc -static-libstdc++
endif

LIBS   = -pthread

NATIVEFLAGS   = -march=native

native:
	$(CC) $(SRC) $(RFLAGS) $(LIBS) $(NATIVEFLAGS) -o $(EXE)-native$(EXT)
debug:
	$(CC) $(SRC) $(RFLAGS) $(LIBS) $(NATIVEFLAGS) -g -o $(EXE)-debug$(EXT)