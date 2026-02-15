
# target 
TARGET_NAME=$(notdir $(shell pwd) )

# flags
CPPFLAGS+=-g
LDFLAGS+=
LDLIBS+=-lm

# OpenCL flags
CPPFLAGS+=-D CL_HPP_TARGET_OPENCL_VERSION=300 
LDLIBS+=$(shell pkgconf --libs OpenCL)

# files
HDRFILES=$(wildcard *.h)
SRCFILES=$(wildcard *.cpp)
OBJFILES=$(addsuffix .o, $(basename $(SRCFILES)))	

# kernels
SRCKERNELS=$(wildcard *.cl)
SPVKERNELS=$(addsuffix .spv, $(basename $(SRCKERNELS)))

LLVM2SPIRV=$(notdir $(word 2, $(shell whereis -b -g llvm-spirv* )))

# detect opencv lib
OPENCVPKG=$(shell pkgconf --list-package-names | grep opencv )

CPPFLAGS+=$(shell pkgconf --cflags $(OPENCVPKG))
LDFLAGS+=$(shell pkgconf --libs-only-L $(OPENCVPKG))
LDLIBS+=$(shell pkgconf --libs-only-l $(OPENCVPKG))

# detect clang
CLANGBIN=$(word 2, $(shell whereis -b clang ))

# build

all: check_opencv check_llvm check_clang $(TARGET_NAME)

check_llvm:
ifeq ($(LLVM2SPIRV),)
	@echo llvm-spirv* not found!
	@echo Try: 'apt-cache search llvm-spirv'
	@echo Try: 'apt install llvm-spirv-*'
	@exit 1
endif

check_opencv:
ifeq ($(OPENCVPKG),)
	@echo OpenCV lib not found!
	@echo Try: 'apt install libopencv-dev'
	@exit 1
endif

check_clang:
ifeq ($(CLANGBIN),)
	@echo CLANG not found.
	@echo Try: 'apt install clang'
	@exit 1
endif

# compile source codes
%.o: %.cpp $(HDRFILES)
	g++ $(CPPFLAGS) -c $< -o $@

# build kernels
%.spv: %.cl $(HDRFILES)
	@echo "---------- kernel >>>>>>>>>>"
	clang -cl-std=CLC++ -target spirv64 -emit-llvm  -c $< -o $<.bc
	$(LLVM2SPIRV) $<.bc -o $@
	@echo "---------- kernel <<<<<<<<<<"

# build app
$(TARGET_NAME): $(SPVKERNELS) $(OBJFILES) $(HDRFILES)
	@echo "---------- app >>>>>>>>>>"
	g++ $(CPPFLAGS) $(LDFLAGS) $(OBJFILES) $(LDLIBS) -o $@
	@echo "---------- app <<<<<<<<<<"

clean:
	rm -f *.o *.bc *.spv $(TARGET_NAME)


