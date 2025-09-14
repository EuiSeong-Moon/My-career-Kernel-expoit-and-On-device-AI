// sample.c : openat()으로 파일을 열고, execve()로 /bin/ls 실행
#define _GNU_SOURCE
#include <fcntl.h>      // openat, O_RDONLY
#include <unistd.h>     // read, close, execve
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(void) {
    const char *target = "/etc/hostname";

    // 1) openat()로 파일 열기 (AT_FDCWD: 현재 작업 디렉터리를 기준으로 path 해석)
    int fd = openat(AT_FDCWD, target, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "openat(%s) failed: %s\n", target, strerror(errno));
        return 1;
    }

    // 2) 간단히 읽어서 화면에 보여주기 (확인용)
    char buf[256] = {0};
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    if (n < 0) {
        fprintf(stderr, "read(%s) failed: %s\n", target, strerror(errno));
        close(fd);
        return 1;
    }
    printf("[sample] read %zd bytes from %s: %s\n", n, target, buf);
    close(fd);

    // 3) execve()로 /bin/ls 실행 (자신의 프로세스를 /bin/ls로 교체)
    //    argv는 NULL-terminated 배열이어야 하고, envp도 NULL-terminated 필요.
    const char *prog = "/bin/ls";
    char *const argv[] = { "ls", "-l", "/", NULL };
    extern char **environ;  // 현재 환경변수 재사용

    printf("[sample] execve(%s -l /)\n", prog);
    fflush(stdout);

    if (execve(prog, argv, environ) == -1) {
        fprintf(stderr, "execve(%s) failed: %s\n", prog, strerror(errno));
        return 1;
    }

    // execve가 성공하면 아래 코드는 실행되지 않습니다.
    return 0;
}

