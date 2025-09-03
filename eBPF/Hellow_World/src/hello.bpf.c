// src/hello.bpf.c
// 간단한 tracepoint 핸들러: execve 진입 시 "Hello, BPF!"를 프린트

#include <linux/types.h>
#include <linux/in.h>
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

// 라이선스 명시 (GPL이 아니면 일부 helper를 못 쓸 수 있음)
char LICENSE[] SEC("license") = "GPL";

// tracepoint용 올바른 시그니처: ctx 하나만!
SEC("tracepoint/syscalls/sys_enter_execve")
int BPF_PROG(hello, const char *filename, const char *const *argv, const char *const *envp) { 
// 현재 프로세스 PID 가져오기
 __u64 pid_tgid = bpf_get_current_pid_tgid();
  __u32 pid = pid_tgid >> 32;
   // bpf_printk는 /sys/kernel/debug/tracing/trace_pipe 로 출력됨
    bpf_printk("Hello, BPF! pid=%d filename=%s\n", pid, filename);
     return 0;
      }
