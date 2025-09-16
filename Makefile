# Compiler and flags
NVCC        := nvcc

# Target
TARGET      := nbody

# Sources
SRC         := nbody.cu

# Default rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $@ $^

# Clean up
clean:
	rm -f $(TARGET) *.o

# Run simulation
run: $(TARGET)
	./$(TARGET)
