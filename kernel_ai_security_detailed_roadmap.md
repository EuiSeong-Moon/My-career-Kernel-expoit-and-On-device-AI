
# 🧠 커널 익스플로잇 탐지 + On-device AI Security 전문가 로드맵 (8+6주)

---

## 📍 1~8주: 커널 익스플로잇 탐지 전문가 실습 로드맵

### ✅ Week 1: UAF 취약점 구조 및 PoC 분석
- 커널 메모리 모델 및 SLUB 구조 학습
- CVE-2017-1000112 PoC 실행 및 KASAN 로그 해석
- gdb/crash로 call trace 분석 및 함수 흐름 도식화
- Markdown 실습 보고서 작성 (PoC 설명, 트리거 조건, 로그 분석 포함)

### ✅ Week 2: Heap Overflow 및 Race Condition
- CVE-2022-0185 heap overflow PoC 분석
- Heap spray 개념 및 SLUB freelist 이해
- Race condition 유도 및 double free 탐지
- PoC 2개 실행 후 크래시 로그 비교 분석

### ✅ Week 3: eBPF 및 kprobe 기초 이해
- eBPF 구조, verifier, map 이해
- bpftrace 설치 및 kprobe로 syscall 후킹 (open, execve 등)
- syscall 인자 값 추적 및 출력
- 작은 탐지기 prototype 제작

### ✅ Week 4: eBPF 기반 익스플로잇 런타임 탐지기 제작
- SLUB free 시점 후킹하여 use-after-free 감지
- refcount 이상 여부 탐지 eBPF 코드 작성
- 탐지 결과를 syslog 또는 json 파일로 출력
- 탐지 정확도/오탐 사례 분석

### ✅ Week 5: Syzkaller 설정 및 커널 Fuzzing
- Android/Linux 커널 KASAN 활성화 후 빌드
- Syzkaller 환경 구성 및 syz-manager.yaml 작성
- fuzzing 실행 → crash 발생 확인
- crash reproduction 스크립트 생성

### ✅ Week 6: Crash Triage 및 분석 자동화
- syz-crasher로 crash 재현
- crash 로그에서 RIP, instruction pointer, call trace 분석
- root cause 파악 + 유사 PoC 작성
- triage 결과 보고서 작성

### ✅ Week 7: CVE 및 보안 패치 분석
- 공개된 CVE 패치 전후 커널 diff 비교
- patch가 막은 exploit 흐름 재현
- 유사 취약점 패턴 도출 및 eBPF 탐지기 적용 가능성 확인

### ✅ Week 8: LSM / SELinux + 전체 결과 정리
- LSM hook 흐름 및 SELinux 정책 개요 학습
- 간단한 SELinux 정책 삽입 실습
- 8주 결과물 정리 (PoC 실행 리포트, 탐지기 코드, 보고서)
- GitHub 업로드용 포트폴리오 제작

---

## 🚀 9~14주: On-device AI Security 확장 로드맵

### ✅ Week 9: On-device AI 시스템 구조 및 위협 모델
- TFLite, NNAPI, CoreML 등 runtime 구조 학습
- 스마트폰/디바이스에서 모델이 어떻게 load되고 execute되는지 분석
- Model stealing, adversarial attack, model tampering 등의 위협 시나리오 학습

### ✅ Week 10: 모델 보호 및 무결성 검증
- 모델 파일 암호화 및 서명, 무결성 해시 비교
- Secure Boot + TrustZone 기반의 모델 보호 원리 학습
- Python으로 TFLite/ONNX 모델 서명 검증 실습

### ✅ Week 11: 추론 syscall 추적 및 탐지기 개발
- AI inference를 수행하는 syscall (mmap, ioctl 등) 흐름 분석
- eBPF를 사용하여 모델 로딩 또는 추론 syscall 감시
- Adversarial input 시도 탐지 룰 설계

### ✅ Week 12: 런타임 변조 및 sandbox 회피 방지
- LD_PRELOAD, debugger attach, dynamic injection 분석
- Anti-debug 기법 적용, SELinux context 보호 실습
- AI 서비스별 신뢰 경계 설정과 보호 대상 도출

### ✅ Week 13: 개인정보 보호 및 output 검증
- 모델 output에서 개인정보 유출 여부 평가
- logcat/crash 로그에서 inference 힌트 제거
- Differential Privacy 및 output obfuscation 개념 적용

### ✅ Week 14: 최종 프로젝트 및 보고서화
- eBPF 기반 AI 추론 syscall 탐지기 개발
- 모델 integrity watchdog 구현
- 전체 시스템 흐름 아키텍처 설계 + 최종 리포트 작성
- GitHub + PDF 포트폴리오화

---

## 🧠 커리어 전략 요약

> 커널 익스플로잇 방어를 기반으로 On-device AI 보안 전문가로 확장 →  
> 희소성과 실전성, 트렌드 수요 모두 갖춘 커리어 완성

