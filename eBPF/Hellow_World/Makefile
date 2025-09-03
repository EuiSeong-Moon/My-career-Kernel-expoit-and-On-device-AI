# Makefile (in ~/hello-ebpf)
BPF_CLANG ?= clang
BPF_LLVM_STRIP ?= llvm-strip
CFLAGS_USR ?= -O2 -g
CFLAGS_BPF ?= -O2 -g -target bpf

INCLUDES := -I./src -I/usr/include -I/usr/include/bpf

SRC_DIR := src
BPF_PROG := $(SRC_DIR)/hello.bpf.c
BPF_OBJ  := $(SRC_DIR)/hello.bpf.o
SKEL_HDR := $(SRC_DIR)/hello.skel.h
USR_SRC  := $(SRC_DIR)/hello.c
USR_BIN  := hello

.PHONY: all clean run logs

all: $(USR_BIN)

$(BPF_OBJ): $(BPF_PROG) $(SRC_DIR)/vmlinux.h
	$(BPF_CLANG) $(CFLAGS_BPF) $(INCLUDES) -c $< -o $@
	$(BPF_LLVM_STRIP) -g $@

$(SKEL_HDR): $(BPF_OBJ)
	# bpftool로 스켈레톤 생성
	bpftool gen skeleton $< > $@

$(USR_BIN): $(USR_SRC) $(SKEL_HDR)
	$(CC) $(CFLAGS_USR) $(USR_SRC) -o $@ -lbpf -lelf -lz

run: all
	@echo "==> Run as root for attaching BPF"
	sudo ./$(USR_BIN)

logs:
	@echo "==> Tail trace_pipe (Ctrl+C to stop)"
	sudo sh -c 'test -d /sys/kernel/debug || mount -t debugfs debugfs /sys/kernel/debug'
	sudo sh -c 'tail -f /sys/kernel/debug/tracing/trace_pipe'

clean:
	rm -f $(BPF_OBJ) $(SKEL_HDR) $(USR_BIN)

