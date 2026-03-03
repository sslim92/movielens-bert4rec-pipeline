# 🎬 MovieLens-32M 기반 영화 추천 시스템

> **6인 팀 프로젝트 (2주)** | Azure Databricks 환경에서 대규모 데이터 처리 및 BERT4Rec 기반 순차적 추천 모델 개발

---

## 📌 프로젝트 개요

MovieLens-32M 및 IMDB 데이터셋을 활용하여 사용자의 영화 시청 이력 기반 순차적 추천 시스템을 구축한 6인 팀 프로젝트(2주)입니다.  
Apache Spark(Databricks)로 대용량 데이터를 처리하고, BERT4Rec 모델에 장르 임베딩을 결합한 커스텀 아키텍처를 개발하였습니다.  
본인은 **영화 추천 모델 고도화** 및 **MLOps 환경 구성**을 담당하여, NDCG@10 기준 **0.7689 → 0.8887(+15.6%)** 성능 향상과 MLflow 기반 배포 파이프라인 구축을 달성했습니다.

---

## 👤 본인 담당 역할

| 영역 | 내용 |
|------|------|
| **데이터 파이프라인** | PySpark를 활용한 Medallion Architecture 기반 데이터 처리 전과정 |
| **추천 모델 고도화** | BERT4Rec + 장르 임베딩 확장 모델 구현 · 버그 수정 · Negative Sampling 개선으로 NDCG@10 15.6% 향상 |
| **MLOps 파이프라인 구축** | MLflow Model Registry 등록 · 의존성 불일치 해결 · Flask 서비스 배포 |

---

## 🗂️ 프로젝트 구조 (담당 파트)

```
📦 MS_DS_Project1
├── 📓 데이터 처리.ipynb            # 데이터 파이프라인 (Bronze → Silver → Gold)
└── 📁 1dt048/
    ├── 📓 bert4rec-with-genres-embed.ipynb       # 커스텀 BERT4Rec 모델 학습
    ├── 📓 rertv4rec_model_mlflow_upload.ipynb    # MLflow 모델 등록
    └── 📓 mlflow_model_load_test.ipynb           # 모델 로드 및 추론 검증
```

---

## 🔧 기술 스택

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=flat&logo=apachespark&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat&logo=databricks&logoColor=white)

- **플랫폼**: Azure Databricks, Unity Catalog
- **데이터 처리**: PySpark, Pandas
- **딥러닝**: PyTorch (Transformer, Multi-Head Attention)
- **모델 관리**: MLflow (Model Registry, pyfunc)
- **데이터셋**: MovieLens-32M, IMDB title.basics

---

## 📊 1단계: 데이터 파이프라인 (`데이터 처리.ipynb`)

### Medallion Architecture

```
[Bronze]  Unity Catalog에서 원시 데이터 로드
    ↓     ratings, movies, tags, links (MovieLens-32M)
    ↓     title_basics (IMDB)
[Silver]  결측치 제거 / 중복 제거 / 평점 범위 필터링 (0~5)
    ↓     영화 제목에서 연도 추출 (정규식)
    ↓     IMDB 데이터로 연도 결측치 보완 (links ↔ title_basics JOIN)
[Gold]    품질 기준 적용 → Train / Validation / Test 분할 → 저장
```

### 주요 처리 내용

**1. 연도 결측치 보완**
- `movies.title` 컬럼에서 정규식으로 연도 추출
- 추출 실패 시 `links.imdbId` → `tconst` 변환 후 IMDB `title_basics.startYear`로 보완
- 잔여 결측치는 수동 매핑으로 처리

**2. 데이터 필터링 기준**
- 영화별 평점 수 **10개 미만** 제외
- 사용자별 리뷰 수 **20개 미만** 제외

**3. Train / Validation / Test 분할 전략**
- **Cold Start** 방식 채택: 사용자 단위 랜덤 분할 (6 : 2 : 2)
- Databricks Unity Catalog `1dt_team8_databricks.final` 스키마에 최종 저장

---

## 🤖 2단계: BERT4Rec + 장르 임베딩 모델 (`1dt048/`)

### 모델 개요

[BERT4Rec(2019)](https://arxiv.org/pdf/1904.06690) 아키텍처를 기반으로, **장르 정보를 임베딩 레이어에 통합**하는 방식으로 확장했습니다.

> 원본 참고: [constantfear/bert4rec](https://github.com/constantfear/bert4rec)

### 커스텀 아키텍처

```
 입력 시퀀스 (영화 ID)
        │
        ├── TokenEmbedding        # 영화 ID → 벡터
        ├── PositionalEmbedding   # 위치 정보 → 벡터
        └── GenresEmbedding       # 장르 원-핫 → 2-layer MLP → 벡터
                 │
        BERTEmbedding (세 임베딩 합산)
                 │
        Transformer Encoder Blocks (×2)
          └─ Multi-Head Self-Attention (heads=2)
          └─ Position-wise Feed-Forward
          └─ Layer Norm + Residual
                 │
        Linear Output → 추천 아이템 예측
```

**GenresEmbedding 구조**
```python
Linear(18 → 36) → ReLU → Linear(36 → hidden_units)
```
- 18개 장르 카테고리를 영화 임베딩과 동일한 차원(256)으로 변환
- 아이템 임베딩, 위치 임베딩과 합산하여 컨텍스트 강화

### 학습 설정 (config)

| 파라미터 | 값 |
|---------|-----|
| max_len | 80 |
| hidden_units | 256 |
| num_heads | 2 |
| num_layers | 2 |
| dropout_rate | 0.1 |
| learning_rate | 0.001 |
| batch_size | 64 |
| num_epochs | 10 |
| mask_prob | 0.15 |

### MLM(Masked Language Model) 방식 학습

입력 시퀀스의 **15%** 토큰을 무작위 선택 후:
- **80%** → `[MASK]` 토큰으로 대체
- **10%** → 랜덤 아이템으로 대체
- **10%** → 원본 유지 (오염 방지)

장르 시퀀스도 동일한 방식으로 변환하여 아이템-장르 일관성 유지

### 데이터 전처리 및 학습 데이터 구성

- **평점 필터링**: 평점 4.0 이상의 양질 상호작용 데이터만 학습에 활용
- **시계열 분할**: 사용자별 시청 이력을 타임스탬프 기준으로 정렬 후 Train / Test 분할 적용

### 성능 개선 과정

기존 BERT4Rec 구현의 문제점을 발견하고 다음 두 가지 핵심 수정을 통해 추천 정확도를 크게 향상시켰습니다.

**1. Output Layer 크기 버그 수정**
- 출력 레이어의 크기 설정 오류를 디버깅하여 전체 아이템 어휘(vocabulary)에 대한 예측이 올바르게 이루어지도록 수정

**2. Negative Sampling 로직 개선**
- 기존: Negative Sample로 선택된 아이템의 장르를 무작위로 할당
- 개선: 실제 대체 아이템의 장르 정보를 그대로 활용하여 아이템-장르 일관성 보장
- 모델이 장르 임베딩을 의미 있게 학습할 수 있도록 개선

### 성능 결과

| 지표 | 개선 전 | 개선 후 | 변화율 |
|------|---------|---------|--------|
| **NDCG@10** | 0.7689 | 0.8887 | **+15.6%** |

---

## 🚀 3단계: MLflow 모델 관리

### 모델 등록 (`rertv4rec_model_mlflow_upload.ipynb`)
- 학습된 BERT4Rec 모델을 `mlflow.pyfunc` 포맷으로 래핑
- MLflow Model Registry에 `bert4rec_v4` 이름으로 등록하여 버전 관리 체계화
- 아이템 인코더/디코더(`.pkl`) 포함하여 end-to-end 추론 가능한 형태로 패키징

### 배포 환경 의존성 문제 해결
- Databricks 서빙 환경과 학습 환경 간 **Python 버전 및 라이브러리 의존성 불일치**로 인한 배포 실패 발생
- MLflow 로그 분석을 통해 충돌 패키지를 특정하고 `conda.yaml` / `requirements.txt` 수정으로 해결
- 최종적으로 **Flask 기반 서비스에 모델을 성공적으로 탑재**하여 실시간 추천 API 제공

### 추론 테스트 (`mlflow_model_load_test.ipynb`)
```python
# 사용자의 시청 이력으로 영화 추천
inference_data = pd.DataFrame({
    "movie_history": [[4896, 1, 4993, 5952, 33794, ...]]
})
recommendations = loaded_model.predict(inference_data)
```

---

## 📈 데이터셋 규모

| 데이터 | 규모 |
|--------|------|
| MovieLens-32M ratings | 약 3,200만 건 |
| MovieLens movies | 87,000+ 편 |
| IMDB title.basics | 수백만 건 (연도 보완용) |
| 최종 학습 사용자 수 | 필터링 후 약 20만+ 명 |

---

## 🌐 전체 팀 프로젝트 구조

```
📦 MS_DS_Project1
├── 📓 데이터 처리.ipynb         ← 담당 (데이터 파이프라인)
├── 📁 1dt003/  XGBoost 추천 모델
├── 📁 1dt011/  ALS 협업 필터링 & 군집화
├── 📁 1dt026/  ALS + Content-Based Hybrid / Flask API
└── 📁 1dt048/                  ← 담당 (BERT4Rec 모델)
```

