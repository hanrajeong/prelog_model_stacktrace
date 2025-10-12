# 금융권 로그 분석기 개발 보고서 - Part 1: 데이터 준비

## 📊 1. 데이터 수집 및 준비

### 1.1 데이터 소스

#### 다중 소스 통합 데이터셋
프로젝트의 현실성과 다양성을 위해 **7개의 서로 다른 소스**에서 데이터를 수집하고 통합했습니다.

**총 샘플**: 22,898개  
**예외 타입**: 50개 클래스 (병합 전) → 44개 (병합 후)

#### 소스별 상세 분포
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source                  | 샘플 수  | 비율   | 설명
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
synthetic_v3_finance    | 6,877개  | 30.0%  | 금융권 도메인 특화 합성 로그
                        |          |        | (로그인, 결제, 정산 등)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
github_actions          | 6,013개  | 26.3%  | GitHub Actions CI/CD 실패 로그
                        |          |        | (실제 오픈소스 프로젝트)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
synthetic_v2            | 5,910개  | 25.8%  | 일반 애플리케이션 합성 로그
                        |          |        | (다양한 예외 패턴)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
hard_negative           | 1,969개  | 8.6%   | 분류 어려운 케이스
                        |          |        | (모델 성능 테스트용)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
bears                   | 1,923개  | 8.4%   | Bears 벤치마크 (BugsJar)
                        |          |        | (실제 Java 프로젝트 버그)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
stackoverflow           |   117개  | 0.5%   | Stack Overflow Q&A
                        |          |        | (개발자 실제 질문)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
real_logs               |    89개  | 0.4%   | 실제 프로덕션 로그
                        |          |        | (익명화 처리)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
총계                    | 22,898개 | 100%   |
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### 데이터 소스 선정 이유

**1. synthetic_v3_finance (30%, 최다)**
```
목적: 금융권 도메인 특화
특징:
✅ 은행/카드사 실제 시나리오 반영
✅ 로그인, 결제, 정산, 대출 등 도메인 커버
✅ 현실적인 스택트레이스 패턴
✅ 다층 Caused by 구조

예시 도메인:
- 로그인/인증
- 결제 처리
- 정산 시스템
- 대출 심사
- 포인트 적립
```

**2. github_actions (26%, 2위)**
```
목적: 실제 CI/CD 환경 로그
특징:
✅ 실제 GitHub 오픈소스 프로젝트
✅ 빌드 실패, 테스트 실패 로그
✅ Gradle, Maven, JUnit 오류
✅ 타임스탬프, 경로 포함

대표 프로젝트:
- Hibernate ORM
- Spring Framework
- Apache Kafka
- Kotlin 프로젝트
```

**3. synthetic_v2 (26%, 3위)**
```
목적: 다양한 예외 패턴 커버
특징:
✅ 일반적인 Java 애플리케이션 오류
✅ NullPointerException, IOException 등
✅ 다양한 프레임워크 (Spring, Hibernate)
✅ 균형잡힌 클래스 분포
```

**4. hard_negative (9%)**
```
목적: 모델 robustness 테스트
특징:
✅ 비슷한 예외 구분 (SQLException vs SQLSyntaxError)
✅ 중첩된 복잡한 스택트레이스
✅ 모호한 에러 메시지
✅ 모델 한계 파악용
```

**5. bears (8%)**
```
목적: 학술적 벤치마크
출처: Bears (BugsJar) 벤치마크
특징:
✅ 실제 버그 데이터
✅ 학술 연구에서 검증됨
✅ 다양한 오픈소스 프로젝트
✅ 비교 연구 가능
```

**6. stackoverflow (0.5%)**
```
목적: 개발자 실제 질문 패턴
특징:
✅ 실제 개발자가 겪는 문제
✅ 다양한 컨텍스트
✅ 소수 샘플 (희귀 케이스)
```

**7. real_logs (0.4%)**
```
목적: 실제 프로덕션 환경
특징:
✅ 익명화된 실제 로그
✅ 프로덕션 특유의 패턴
✅ 검증 및 테스트용
```

### 1.2 데이터 수집 과정

#### Step 1: 다중 소스 데이터 수집

**1.1 금융권 합성 데이터 생성 (synthetic_v3_finance)**
```python
# 금융 도메인 특화 로그 생성
domains = [
    '로그인/인증',
    '결제 처리', 
    '정산 시스템',
    '대출 심사',
    '포인트 적립',
    '인증서버',
    'API 통신'
]

banks = [
    'KB국민은행', '신한은행', '우리은행',
    'IBK기업은행', '하나은행', 'NH농협',
    '삼성카드', '현대카드', '롯데카드'
]

# 현실적인 스택트레이스 패턴 생성
# - 3-7 레이어 스택
# - Caused by 체인
# - 금융권 패키지 구조 (com.{bank}.{domain})
```

**1.2 GitHub Actions 로그 크롤링**
```python
# GitHub Actions API를 통한 실패 로그 수집
repositories = [
    'hibernate/hibernate-orm',
    'spring-projects/spring-framework',
    'apache/kafka',
    'JetBrains/kotlin',
    # ... 100+ 프로젝트
]

# CI/CD 실패 로그 추출
# - 빌드 실패
# - 테스트 실패
# - 의존성 오류
```

**1.3 일반 합성 데이터 (synthetic_v2)**
```python
# 다양한 예외 패턴 생성
frameworks = ['Spring', 'Hibernate', 'JPA', 'MyBatis']
exception_types = [
    'NullPointerException',
    'IOException',
    'SQLException',
    'ClassNotFoundException',
    # ... 44개 타입
]

# 균형잡힌 클래스 분포 유지
```

**1.4 Hard Negative 케이스 생성**
```python
# 모델이 혼동하기 쉬운 케이스
confusing_pairs = [
    ('SQLException', 'SQLSyntaxErrorException'),
    ('IOException', 'FileNotFoundException'),
    ('IllegalStateException', 'IllegalArgumentException'),
    # ...
]

# 복잡한 중첩 스택트레이스
# 모호한 에러 메시지
```

**1.5 Bears 벤치마크 수집**
```bash
# Bears (BugsJar) 데이터셋
data_collection/temp_bugsjar/
├── traccar-traccar/
├── FasterXML-jackson-databind/
├── spring-projects/
└── ... (검증된 버그 데이터)
```

**1.6 Stack Overflow 크롤링**
```python
# Stack Overflow에서 로그 관련 질문 수집
tags = ['java', 'exception', 'stacktrace', 'error-handling']
min_votes = 5  # 검증된 질문만

# 실제 개발자가 겪는 문제 패턴
```

**1.7 실제 프로덕션 로그 (익명화)**
```python
# 실제 서비스 로그 (소수)
# - 개인정보 제거
# - IP, 사용자명 마스킹
# - 검증 및 테스트용
```

#### Step 2: 데이터 통합 및 정제
```python
def integrate_multi_source_data():
    """
    7개 소스의 데이터를 통합하고 정제
    """
    all_data = []
    
    # 1. 각 소스별 로드
    synthetic_v3 = load_synthetic_v3_finance()
    github = load_github_actions()
    synthetic_v2 = load_synthetic_v2()
    hard_neg = load_hard_negatives()
    bears = load_bears_benchmark()
    stackoverflow = load_stackoverflow()
    real = load_real_logs()
    
    # 2. 포맷 통일
    for data in [synthetic_v3, github, ...]:
        normalized = normalize_format(data)
        all_data.extend(normalized)
    
    # 3. 중복 제거
    unique_data = remove_duplicates(all_data)
    
    # 4. 품질 검증
    validated = validate_quality(unique_data)
    
    return validated
```

#### Step 3: 데이터 검증
```python
def validate_data(data):
    """품질 검증"""
    checks = [
        # 필수 필드 존재
        has_required_fields(data),
        
        # 스택트레이스 유효성
        is_valid_stacktrace(data['input']),
        
        # 예외 타입 정규화
        is_normalized_exception_type(data['exception_type']),
        
        # 분석 내용 존재
        has_analysis(data['analysis']),
        
        # 인코딩 문제 없음
        is_valid_encoding(data)
    ]
    
    return all(checks)
```

### 1.3 데이터 구조

#### 최종 데이터 포맷 (JSONL)
```json
{
  "input": "Exception in thread \"main\" java.lang.NullPointerException: Cannot invoke \"User.getName()\" because \"user\" is null\n    at com.example.service.UserService.getUser(UserService.java:45)\n    at com.example.controller.UserController.handleRequest(UserController.java:23)",
  
  "exception_type": "NullPointerException",
  
  "analysis": "type=NullPointerException; cause=user 객체가 null입니다.\n\n📍 문제 위치:\n  - 파일: UserService.java\n  - 라인: 45\n  - 메서드: getUser\n\n🔧 수정 방법:\nif (user == null) {\n    throw new IllegalArgumentException(\"User not found\");\n}\nString name = user.getName();"
}
```

### 1.4 데이터 분포 분석

#### 클래스별 샘플 수
```
📊 상위 10개 클래스 (전체의 51%):
1. IOException                        2,153개 (9.4%)
2. NullPointerException              1,901개 (8.3%)
3. ArithmeticException               1,123개 (4.9%)
4. ArrayIndexOutOfBoundsException    1,119개 (4.9%)
5. IllegalStateException             1,063개 (4.6%)
6. IllegalArgumentException            987개 (4.3%)
7. SQLException                        969개 (4.2%)
8. HttpClientErrorException            910개 (4.0%)
9. AssertionError                      798개 (3.5%)
10. SocketTimeoutException             734개 (3.2%)

📊 하위 10개 클래스 (전체의 4%):
41. org.opentest4j.AssertionFailedError                   130개 (0.6%)
42. org.apache.activemq.transport.TransportDisposedIOE    128개 (0.6%)
43. org.apache.flink.client.program.ProgramInvocation     101개 (0.4%)
44. PASSED                                                100개 (0.4%)
45. org.apache.flink.runtime.JobException                  99개 (0.4%)
46. RuntimeException                                       97개 (0.4%)
47. WorkExecutionException                                 95개 (0.4%)
48. org.jetbrains.kotlin.gradle.tasks.FailedCompilation    91개 (0.4%)
49. org.apache.flink.runtime.client.JobExecutionExcepti    86개 (0.4%)
50. org.apache.beam.vendor.guava...UncheckedExecutionE     86개 (0.4%)
```

### 1.5 데이터 불균형 문제 및 해결

#### 문제점
- **클래스 불균형 심각**: 상위 클래스와 하위 클래스 최대 25배 차이
- **Long-tail 분포**: 일부 클래스는 100개 미만의 샘플
- **모델 학습 어려움**: 희귀 클래스에 대한 학습 부족

#### 해결 방안 1: 클래스 병합
```python
# merge_similar_classes.py
MERGE_RULES = {
    # Flink 관련 예외 통합 (3개 → 1개)
    'org.apache.flink.client.program.ProgramInvocationException': 'FlinkException',
    'org.apache.flink.runtime.JobException': 'FlinkException',
    'org.apache.flink.runtime.client.JobExecutionException': 'FlinkException',
    
    # 빌드 관련 예외 통합 (4개 → 1개)
    'ModelBuildingException': 'BuildException',
    'org.gradle.api.internal.exceptions.MarkedVerificationException': 'BuildException',
    'org.jetbrains.kotlin.gradle.tasks.FailedCompilationException': 'BuildException',
    'org.gradle.api.tasks.TaskExecutionException': 'BuildException',
    
    # 테스트/Assertion 통합 (3개 → 1개)
    'org.opentest4j.AssertionFailedError': 'AssertionError',
    'java.lang.AssertionError': 'AssertionError',
    
    # Transport 예외 통합
    'org.apache.activemq.transport.TransportDisposedIOException': 'TransportException',
    'org.apache.activemq.transport.InactivityIOException': 'TransportException',
}

# 결과
원본 클래스: 50개
병합 후 클래스: 44개 (12% 감소)
병합된 샘플: 1,817개
```

#### 해결 방안 2: Focal Loss 적용
```python
class FocalLoss(nn.Module):
    """
    클래스 불균형 문제 해결을 위한 Focal Loss
    - 쉬운 샘플의 가중치 감소
    - 어려운 샘플에 집중
    """
    def __init__(self, alpha=None, gamma=2.0):
        self.alpha = alpha  # 클래스별 가중치
        self.gamma = gamma  # Focusing 파라미터
```

#### 해결 방안 3: Class Weighting
```python
# 클래스별 가중치 계산
class_counts = Counter(labels)
total = len(labels)
class_weights = {
    cls: total / (len(class_counts) * count)
    for cls, count in class_counts.items()
}

# 희귀 클래스에 더 높은 가중치 부여
```

### 1.6 데이터 분할

#### Train / Validation Split
```
총 샘플: 22,898개
Train: 20,608개 (90%)
Validation: 2,290개 (10%)

분할 방식: Stratified (클래스 비율 유지)
```

### 1.7 데이터 증강 (고려사항)

#### 시도하지 않은 이유
1. **로그의 특수성**: 로그는 정확한 패턴이 중요
2. **의미 보존**: 무작위 증강 시 의미 왜곡 가능
3. **시간 제약**: 기본 학습 우선

#### 향후 개선 방안
- 백트랜슬레이션 (paraphrasing)
- 변수명 랜덤화
- 숫자 값 변경

---

## 📈 2. 데이터 품질 검증

### 2.1 데이터 검증 과정

#### 체크리스트
✅ 모든 샘플에 예외 타입 존재
✅ 스택트레이스 형식 유효성
✅ 중복 제거 완료
✅ 인코딩 문제 없음 (UTF-8)
✅ JSON 파싱 오류 없음

### 2.2 출력 길이 분석

```python
# 출력 토큰 길이 분포
분석:
- 평균 길이: 42 토큰
- 중앙값: 38 토큰
- 최대 길이: 182 토큰
- 99%ile: 145 토큰

결론:
→ max_output_length=256으로 설정 (충분한 여유)
→ T5-small 출력 길이 제한 문제 없음
```

---

## 💾 3. 최종 데이터셋

### 3.1 파일 구조
```
data_collection/collected_logs/
├── with_code_guidance.jsonl          (원본, 50 클래스)
├── with_code_guidance_merged.jsonl   (병합, 44 클래스)
└── with_code_guidance_sampled.jsonl  (샘플링, 10K)

models/
├── logbert_data/
│   ├── train.txt  (LogBERT 포맷, 20,608개)
│   └── test.txt   (LogBERT 포맷, 2,290개)
└── deeplog_data/
    ├── sequences.txt
    └── labels.txt
```

### 3.2 데이터 통계
```
총 샘플 수: 22,898개
평균 입력 길이: 450 토큰
평균 출력 길이: 42 토큰
클래스 수: 44개 (병합 후)
데이터 크기: ~18.6 MB
```

### 3.3 데이터 품질 지표
```
완전성: 100% (모든 필수 필드 존재)
일관성: 98.5% (소수의 특수 케이스 제외)
정확성: 수동 검증 샘플 100개 중 97개 정확
다양성: 50개 프로젝트, 다양한 예외 타입
```

---

## 🎯 4. 데이터 준비의 핵심 인사이트

### 4.1 성공 요인
1. **실제 프로젝트 데이터**: BugsJar의 실제 버그 데이터 사용
2. **구조화된 형식**: 일관된 JSON 포맷
3. **클래스 병합**: 불균형 완화
4. **충분한 샘플 수**: 22,898개로 딥러닝 학습 가능

### 4.2 한계점 및 향후 개선
1. **Java 중심**: Python 등 다른 언어 부족
2. **도메인 한정**: 금융권 특화 데이터 필요
3. **실시간 데이터**: 프로덕션 로그 추가 필요

### 4.3 재현성
```bash
# 전체 데이터 준비 과정 재현
1. BugsJar 데이터셋 다운로드
2. python collect_logs.py
3. python merge_similar_classes.py
4. 검증 및 분할
```
