#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Python + BCC로 eBPF “Hello, BPF!” (execve tracepoint)
#
from bcc import BPF
import resource
import signal
import sys

# 1) MEMLOCK 한도 풀기 (실행 중 쉘에 한정)
resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

# 2) eBPF 프로그램(C) - BCC는 이 문자열을 JIT 컴파일해서 로드한다.
#    TRACEPOINT_PROBE 매크로를 쓰면 tracepoint용 ctx(struct)를 자동으로 맞춰준다.
bpf_program = r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <linux/limits.h>

TRACEPOINT_PROBE(syscalls, sys_enter_execve)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;

    char fname[128] = {};
    // tracepoint args의 filename은 사용자 공간 포인터다 → 안전하게 유저 메모리에서 복사
    bpf_probe_read_user_str(fname, sizeof(fname), args->filename);

    bpf_trace_printk("Hello, BPF! pid=%d filename=%s\n", pid, fname);
    return 0;
}
"""

# 3) 로드 & 어태치
b = BPF(text=bpf_program)

print("Running. Press Ctrl+C to stop.")
print("Tip) sudo cat /sys/kernel/debug/tracing/trace_pipe  (bcc가 아래에서 자동 출력도 해줌)")

# 4) SIGINT 처리
def handle_sigint(signum, frame):
    print("\nStopping...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

# 5) trace_pipe를 tail처럼 출력
#    (커널의 bpf_trace_printk() 출력이 여기로 흘러온다)
b.trace_print()

