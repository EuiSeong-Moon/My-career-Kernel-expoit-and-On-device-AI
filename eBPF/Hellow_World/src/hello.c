// src/hello.c
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <bpf/libbpf.h>

// bpftool로 생성할 스켈레톤 헤더
#include "hello.skel.h"

static volatile sig_atomic_t stop;

static void handle_sigint(int sig) {
    stop = 1;
}

int main(void) {
    struct hello_bpf *skel;
    int err;

    libbpf_set_strict_mode(LIBBPF_STRICT_ALL);

    // BPF FS/tracefs가 안 붙어 있으면 libbpf가 자체 처리하지만,
    // 문제 시 아래 두 줄로 수동 마운트:
    // system("mount -t bpf bpf /sys/fs/bpf 2>/dev/null || true");
    // system("mount -t tracefs nodev /sys/kernel/tracing 2>/dev/null || true");

    skel = hello_bpf__open_and_load();
    if (!skel) {
        fprintf(stderr, "Failed to open/load BPF skeleton\n");
        return 1;
    }

    // Attach: tracepoint에 프로그램 연결
    err = hello_bpf__attach(skel);
    if (err) {
        fprintf(stderr, "Failed to attach BPF programs: %d\n", err);
        hello_bpf__destroy(skel);
        return 1;
    }

    printf("hello-ebpf is running. Press Ctrl+C to exit.\n");
    printf("Tip) Logs: sudo cat /sys/kernel/debug/tracing/trace_pipe\n");

    signal(SIGINT, handle_sigint);
    while (!stop) {
        sleep(1);
    }

    hello_bpf__destroy(skel);
    printf("bye.\n");
    return 0;
}

