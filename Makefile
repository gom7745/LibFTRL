VERSION = 1.0.0.117
CXX = g++
CXXFLAGS = -Wall -g -std=c++0x -march=native

# comment the following flags if you do not want to use OpenMP
DFLAG += -DUSEOMP
CXXFLAGS += -fopenmp

all: train predict ftrl-${VERSION}.zip

train: train.cpp ftrl.o
	$(CXX) $(CXXFLAGS) -o $@ $^

predict: predict.cpp ftrl.o
	$(CXX) $(CXXFLAGS) -o $@ $^

ftrl.o: ftrl.cpp ftrl.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

ftrl-${VERSION}.zip: train predict
	zip $@ $^

clean:
	rm -f train predict ftrl.o *.bin.* ftrl-${VERSION}.zip
