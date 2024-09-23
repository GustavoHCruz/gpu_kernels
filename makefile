# Makefile para compilar e executar código CUDA com vários conjuntos de valores

# Nome do executável
TARGET = output

# Compilador CUDA
NVCC = nvcc

# Flags de compilação
NVCCFLAGS = --std=c++11 -I /usr/include/c++/10 -I /usr/lib/cuda/include/

# Fontes CUDA
CUFILES = reduce_max.cu

# Conjuntos de valores para executar
RUNS = "1000000 30" "16000000 30" "30000000 30" "100000000 30"

# Regras
all: $(TARGET)

$(TARGET): $(CUFILES)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

run:
	@for params in $(RUNS); do \
		echo "Running ./$(TARGET) $$params"; \
		./$(TARGET) $$params; \
	done

clean:
	rm -f $(TARGET)
