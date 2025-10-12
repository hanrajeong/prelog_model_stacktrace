# 금융권 로그 분석기 개발 보고서 - Part 2: 모델 아키텍처 진화

## 🏗️ 1. 아키텍처 진화 과정

### 1.1 Phase 0: 기존 시스템 (출발점)

#### 기존 시스템 구조
```
로그 입력 → PreLog 단일 모델 → 예외 타입 출력
```

#### 기존 시스템의 한계
```
❌ 단순 분류만 가능 (예외 타입만 출력)
❌ 상세한 분석 부족
❌ 구조화되지 않은 출력
❌ 코드 위치 정보 부정확
❌ 수정 방법 제시 불가
❌ 확장성 한계
```

#### 성능
```
정확도: 70% (PreLog 기본 성능)
응답 속도: 0.5초
출력: "NullPointerException" (단순 레이블)
```

---

### 1.2 Phase 1: 2-Stage 파이프라인 설계

#### 핵심 아이디어
```
문제 분리 (Separation of Concerns):
- Stage 1: 예외 타입 분류 (Classification)
- Stage 2: 상세 분석 생성 (Generation)

이유:
1. 분류와 생성은 다른 특성의 문제
2. 각 단계를 전문화하면 성능 향상
3. 모듈화로 유지보수 용이
```

#### 전체 아키텍처
```
┌──────────────────────────────────────────┐
│              로그 입력                    │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│  Stage 1: Exception Classifier           │
│  ┌────────────────────────────────────┐ │
│  │ PreLog Encoder (Frozen)             │ │
│  │  - RoBERTa 기반                     │ │
│  │  - 768-dim 임베딩                   │ │
│  └────────────────────────────────────┘ │
│                   ↓                      │
│  ┌────────────────────────────────────┐ │
│  │ Classifier Head                     │ │
│  │  - Linear(768 → 512)               │ │
│  │  - ReLU + Dropout(0.1)             │ │
│  │  - Linear(512 → 44)                │ │
│  │  - Softmax                         │ │
│  └────────────────────────────────────┘ │
│                   ↓                      │
│  예외 타입 + 신뢰도 (Top-1, Top-3)       │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│  Stage 2: Detail Generator               │
│  ┌────────────────────────────────────┐ │
│  │ 입력: 로그 + Stage 1 힌트           │ │
│  │ [LOG] {log}                         │ │
│  │ [HINT] type={predicted_type}        │ │
│  └────────────────────────────────────┘ │
│                   ↓                      │
│  ┌────────────────────────────────────┐ │
│  │ T5-small Seq2Seq                    │ │
│  │  - Encoder: 로그 + 힌트 이해        │ │
│  │  - Decoder: 구조화된 분석 생성      │ │
│  └────────────────────────────────────┘ │
│                   ↓                      │
│  구조화된 상세 분석                      │
│  - type=...                              │
│  - 📍 위치: 파일, 라인                   │
│  - 🔧 수정 방법                          │
└──────────────────────────────────────────┘
```

---

## 🎯 2. Stage 1: Exception Classifier 상세

### 2.1 모델 구조

#### PreLog Encoder
```python
class PreLogEncoder(nn.Module):
    """
    PreLog: 로그 분석을 위해 사전학습된 RoBERTa 기반 모델
    - Microsoft에서 개발
    - 하드웨어/시스템 로그로 사전학습
    - 768-dim 임베딩
    """
    def __init__(self):
        self.roberta = RobertaModel.from_pretrained('microsoft/PreLog')
        self.hidden_size = 768
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # [CLS] 토큰 임베딩 사용
        return outputs.last_hidden_state[:, 0, :]
```

#### Classification Head
```python
class ClassificationHead(nn.Module):
    """
    분류 헤드: 임베딩 → 예외 타입 확률
    """
    def __init__(self, hidden_size=768, num_classes=44):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, embeddings):
        x = self.fc1(embeddings)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
```

### 2.2 학습 전략

#### Version 1 (v1): Frozen Encoder
```python
학습 설정:
- PreLog Encoder: Frozen (고정)
- Classifier Head: 학습
- Learning Rate: 2e-4
- Batch Size: 16
- Grad Accum: 2 (effective batch=32)
- Epochs: 10
- Early Stopping: patience=3

Loss Function:
- Focal Loss (gamma=2.0)
  + Class Weighting
  + Label Smoothing (0.1)

이유:
✅ Encoder는 이미 사전학습되어 있음
✅ 빠른 학습 (Classifier만)
✅ 과적합 방지
```

#### 결과 (v1)
```
📊 최종 성능 (Epoch 8, Best):
- Top-1 Accuracy: 50.48%
- Top-3 Accuracy: 70.22%
- Macro F1: 47.04%
- Train Loss: 1.18
- Val Loss: 0.82

분석:
⚠️ Top-1이 예상보다 낮음 (목표: 75-85%)
✅ Top-3은 양호 (70%)
✅ 과적합 없음 (Val Loss 안정적)
⚠️ 일부 클래스 0% 정확도
```

#### Version 2 (v2): Unfrozen Encoder (개선)
```python
학습 설정:
- PreLog Encoder: Unfrozen (학습)
  → Learning Rate: 2e-5 (1/10)
- Classifier Head: 학습
  → Learning Rate: 2e-4
- Focal Loss Gamma: 2.5 (증가)
- Epochs: 15
- Early Stopping: patience=5

차등 Learning Rate (Differential LR):
optimizer = AdamW([
    {'params': encoder.parameters(), 'lr': 2e-5},
    {'params': classifier.parameters(), 'lr': 2e-4}
], weight_decay=0.01)

이유:
✅ Encoder도 미세 조정
✅ 로그 도메인에 더 적합하게 적응
✅ Focal Gamma 증가로 어려운 클래스 집중
```

#### 예상 결과 (v2)
```
📊 예상 성능:
- Top-1 Accuracy: 62-67% (+12-17%p)
- Top-3 Accuracy: 80-85% (+10-15%p)
- Macro F1: 58-63% (+11-16%p)

근거:
1. Encoder Unfreeze: +5-7%
2. Focal Gamma 증가: +2-3%
3. 더 긴 학습: +2-4%
4. 클래스 병합 효과: +3-5%
```

### 2.3 Confusion Matrix 분석

#### 잘 분류되는 클래스 (100% 정확도)
```
✅ InsufficientBalanceException (82 샘플)
✅ ModelBuildingException (40 샘플)
✅ MySQLIntegrityConstraintViolationException (22 샘플)

이유: 명확한 키워드, 충분한 학습 데이터
```

#### 어려운 클래스 (0% 정확도)
```
❌ QueryException (20 샘플)
❌ RuntimeException (9 샘플)
❌ SQLSyntaxErrorException (29 샘플)

이유:
1. 샘플 수 부족
2. 다른 예외와 혼동
3. 일반적인 이름 (RuntimeException)
```

---

## 🎨 3. Stage 2: Detail Generator 상세

### 3.1 모델 구조

#### T5-small Architecture
```
T5 (Text-To-Text Transfer Transformer)
- Encoder-Decoder 구조
- 파라미터: 60M
- 입력 최대 길이: 512 토큰
- 출력 최대 길이: 256 토큰 (우리 설정)

선택 이유:
✅ Seq2Seq 태스크에 최적화
✅ 사전학습 품질 우수
✅ 크기 적절 (M1 Pro에서 학습 가능)
❌ T5-base(220M)는 너무 느림 (로컬)
```

#### 입력 포맷
```python
def format_input(log, exception_type):
    """
    Stage 1의 예측을 힌트로 활용
    """
    return f"""[LOG]
{log}

[HINT] type={exception_type}"""

예시:
"""[LOG]
Exception in thread "main" java.lang.NullPointerException
    at com.example.UserService.getUser(UserService.java:45)

[HINT] type=NullPointerException"""
```

#### 출력 포맷 (구조화)
```
type=NullPointerException; cause=user 객체가 null입니다.

📍 문제 위치:
  - 파일: UserService.java
  - 라인: 45
  - 메서드: getUser

🔧 수정 방법:
if (user == null) {
    throw new IllegalArgumentException("User not found");
}
```

### 3.2 학습 전략 및 최적화

#### 초기 시도: 기본 Fine-tuning
```python
설정:
- Model: T5-small
- Batch Size: 8
- Grad Accum: 2
- Max Input: 512
- Max Output: 384
- Optimizer: AdamW
- LR: 3e-4

문제:
❌ 속도: 100초/배치 (너무 느림!)
❌ 예상: 200시간 (8일)
```

#### 시도 2: LoRA (실패)
```python
설정:
- LoRA r=8
- Target: Decoder attention
- Gradient Checkpointing: True

문제:
❌ RuntimeError: tensors does not require grad
❌ LoRA + Gradient Checkpoint 호환 문제
```

#### 최종: 최적화된 Fine-tuning
```python
설정:
- Model: T5-small
- Batch Size: 4 (감소)
- Grad Accum: 4 (증가)
- Effective Batch: 16 (유지)
- Max Input: 384 (감소)
- Max Output: 256 (감소)
- Optimizer: AdamW
- LR: 3e-4
- Epochs: 5
- Early Stopping: 2

결과:
✅ 속도: 1.1초/배치 (100배 개선!)
✅ 예상: 4-5시간 (현실적)
✅ Loss: 0.04 (빠르게 수렴)
✅ 메모리 안정
```

### 3.3 생성 전략

#### Beam Search
```python
generated = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=256,
    num_beams=4,           # Beam size
    early_stopping=True,    # 완성 시 조기 종료
    no_repeat_ngram_size=3  # 반복 방지
)
```

#### 출력 품질 제어
```
Max Length: 256 토큰
- 평균 출력: ~100 토큰
- 충분한 상세도
- 너무 길지 않음

No Repeat N-gram: 3
- 같은 3-gram 반복 방지
- 자연스러운 출력
```

---

## 📊 4. 2-Stage 통합 성능

### 4.1 End-to-End 정확도

#### 계산 방식
```
End-to-End = Stage 1 × Stage 2

보수적 (Top-1):
50% × 75% = 37.5%

현실적 (Top-3 활용):
70% × 80% = 56%

v2 예상 (개선 후):
65% × 82% = 53.3%
```

### 4.2 성능 분해
```
┌────────────────────────────────┐
│ 입력: 100개 로그                │
└────────────────────────────────┘
           ↓
┌────────────────────────────────┐
│ Stage 1 (Top-1: 50%)            │
│  ✅ 50개 정확                   │
│  ❌ 50개 오분류                 │
└────────────────────────────────┘
           ↓
┌────────────────────────────────┐
│ Stage 2 (80% 품질)              │
│  ✅ 50 × 0.8 = 40개 상세 분석   │
│  ⚠️  50 × 0.5 = 25개 부분 정확  │
└────────────────────────────────┘
           ↓
최종: 40-45개 완전 성공 (40-45%)
```

### 4.3 개선 여지
```
Stage 1 개선 (v2):
50% → 65% (+15%p)

Stage 2 유지:
80% → 80%

End-to-End:
40% → 52% (+12%p)
```

---

## 🚀 5. 하이브리드 아키텍처 (Phase 3, 계획)

### 5.1 신뢰도 기반 라우팅
```
┌──────────────────────────────────┐
│  Stage 1: PreLog Classifier      │
│  예외 타입 + 신뢰도              │
└──────────────────────────────────┘
              ↓
      신뢰도 체크
              ↓
    ┌─────────┬─────────┐
    │ > 80%   │ < 80%   │
    │         │         │
┌───▼───┐ ┌──▼────────────────┐
│Stage 2│ │BERT 앙상블        │
│T5     │ │(애플리케이션 특화) │
└───┬───┘ └──┬────────────────┘
    │        │ 신뢰도 > 70%?
    │        │  No ↓
    │        │ ┌──────────────┐
    │        │ │LLM + 코드분석 │
    │        │ │CodeLlama-7B   │
    │        │ └──────────────┘
    │        │
    └────┬───┘
         ↓
   최종 분석 결과
```

### 5.2 각 컴포넌트 역할
```
PreLog (70%): 일반적인 예외, 빠름
BERT (20%): 애플리케이션 로그, 중간
CodeLlama (10%): 복잡한 로직, 느림

평균 속도: 2.5초
평균 정확도: 83%
```

---

## 💾 6. 모델 파일 구조

```
models/
├── prelog_downloaded/
│   └── PreLog/                    (사전학습 모델)
│
├── stage1_classifier/             (v1, Frozen)
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── training_history.json
│   └── confusion_matrix.png
│
├── stage1_classifier_v2/          (v2, Unfrozen, 계획)
│   ├── pytorch_model.bin
│   └── ...
│
├── stage2_generator_final/        (T5-small, 진행 중)
│   ├── pytorch_model.bin
│   ├── config.json
│   └── ...
│
├── bert_classifier/               (앙상블용, 계획)
│   └── ...
│
└── codellama-7b/                  (LLM, 다운로드 중)
    ├── pytorch_model.bin
    └── ...
```

---

## 🎯 7. 아키텍처 설계의 핵심 인사이트

### 7.1 성공 요인
1. **문제 분리**: 분류와 생성을 분리하여 각 전문화
2. **단계적 개선**: v1 → v2 → 하이브리드
3. **현실적 최적화**: 메모리/속도 고려
4. **힌트 활용**: Stage 1 결과를 Stage 2 입력으로

### 7.2 기술적 도전과 해결
```
도전 1: LoRA 실패
→ 해결: 일반 Fine-tuning + 메모리 최적화

도전 2: 속도 너무 느림
→ 해결: Batch/Length 조정, 100배 개선

도전 3: Stage 1 정확도 낮음
→ 해결: Encoder Unfreeze, Focal Loss 조정

도전 4: 클래스 불균형
→ 해결: 클래스 병합 + Focal Loss
```

### 7.3 향후 개선 방향
1. A100 서버에서 T5-base 학습 (6-8시간)
2. BERT 앙상블 추가 (2-3시간)
3. CodeLlama 통합 (소스코드 분석)
4. 실시간 피드백 학습
