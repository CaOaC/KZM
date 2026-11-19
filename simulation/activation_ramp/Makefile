# ====== Compilers ======
NVCC      ?= nvcc
CXX       ?= g++

# ====== GPU Arch (A800 = GA100 → sm_80) ======
ARCH_SM   ?= 80
NVCC_ARCH  = -gencode arch=compute_$(ARCH_SM),code=sm_$(ARCH_SM) \
             -gencode arch=compute_$(ARCH_SM),code=compute_$(ARCH_SM)

# ====== Flags ======
CXXFLAGS  = -O3 -Wall -std=c++17 $(INCLUDES) -MMD -MP
NVCCFLAGS = -O3 -std=c++17 -Xcompiler -Wall -rdc=true $(NVCC_ARCH) $(INCLUDES) -MMD -MP

# 可按需开启调试信息：
# CXXFLAGS  += -g
# NVCCFLAGS += -G -lineinfo

LDFLAGS   = -lcurand
TARGET    = kickModel

# ====== Sources / Objects ======
# 你的工程里列出的对象文件（不含扩展名）
OBJS = main.o Interract.o Cell.o Box.o InputOutput.o Particle.o var.o System.o PullMethod.o

# 根据同名 .cu/.cpp 自动匹配编译规则（存在则编译）
CU_SRCS   = $(patsubst %.o,%.cu,$(OBJS))
CPP_SRCS  = $(patsubst %.o,%.cpp,$(OBJS))

DEPS      = $(OBJS:.o=.d)

# ====== Default ======
.PHONY: all clean
all: $(TARGET)

# ====== Link ======
# 用 nvcc 来链接（处理 rdc 的设备链接）
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# ====== Compile rules ======
# .cu → .o
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# .cpp → .o  （如果项目里有同名 .cpp，会走这条）
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# 依赖文件
-include $(DEPS)

# ====== Clean ======
clean:
	rm -f $(TARGET) $(OBJS) $(DEPS)
