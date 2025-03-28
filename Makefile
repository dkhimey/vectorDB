# Compiler
CXX := g++
CXXFLAGS := -O2 -std=c++17 -Iexternal/hnswlib/hnswlib -Iexternal/KaHIP/interface

# KaHIP settings
KAHIP_DIR := external/KaHIP
KAHIP_LIB := $(KAHIP_DIR)/lib/libKaHIP.a

# Source files
SRCS := pyramid.cpp
OBJS := $(SRCS:.cpp=.o)

# Target executable
TARGET := pyramid

# Default target
all: $(TARGET)

# Compile object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build KaHIP (if needed)
$(KAHIP_LIB):
	cd $(KAHIP_DIR) && mkdir -p build && cd build && cmake .. -DNOMPI=ON && make
	cp $(KAHIP_DIR)/build/libkahip_static.a $(KAHIP_LIB)

# Link the final executable
$(TARGET): $(OBJS) $(KAHIP_LIB)
	$(CXX) $(OBJS) -o $(TARGET) -L$(KAHIP_DIR)/lib -lKaHIP

# Clean build files
clean:
	rm -f $(OBJS) $(TARGET)
	rm -rf $(KAHIP_DIR)/build

.PHONY: all clean
