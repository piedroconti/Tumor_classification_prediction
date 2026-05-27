# Lung Cancer Prediction Project

이 프로젝트는 폐암 데이터에 대해 다음 3가지 예측 문제를 수행합니다.

1. **Task 1: Tumor vs Normal Classification**  
   Tumor/NAT expression 데이터를 보고 `Tumor` 또는 `Normal`을 예측합니다.

2. **Task 2: LUAD vs LSCC Classification**  
   Tumor expression 데이터를 보고 `LUAD` 또는 `LSCC`를 예측합니다.

3. **Task 3: Survival Prediction**  
   Tumor expression 데이터를 보고 `Survival` 또는 `Death`를 예측합니다.

이 버전은 `src/predict.py` 하나로 세 문제를 한 번에 실행하지 않습니다.  
각 문제마다 별도의 실행 파일을 사용합니다.

```text
src/predict_tumor_normal.py   # Task 1
src/predict_luad_lscc.py      # Task 2
src/predict_survival.py       # Task 3
```

또한 예측 결과를 CSV/TSV 파일로 저장하지 않고, **터미널에 바로 출력**합니다.

출력에는 채점에 필요한 예측 결과 중심의 column만 표시합니다. 로컬 검증용 보조 정보인 `tissue_type`이나 `input_cancer_file`은 출력하지 않습니다.

---

## 1. 프로젝트 구조

```text
lung_cancer_prediction_taskwise_terminal/
├── README.md
├── requirements.txt
├── models/
│   ├── lung_cancer_models.joblib
│   └── lung_cancer_models.summary.json
├── src/
│   ├── common.py
│   ├── predict_tumor_normal.py
│   ├── predict_luad_lscc.py
│   ├── predict_survival.py
│   └── train_model.py
└── scripts/
    └── check_public_training_accuracy.py
```

`models/lung_cancer_models.joblib`에는 공개 training TSV로 미리 학습한 모델이 저장되어 있습니다.  
따라서 평가자는 training TSV 없이 평가용 TSV만 넣고 예측을 실행할 수 있습니다.

---

## 2. 설치 방법

```bash
pip install -r requirements.txt
```

필요한 패키지는 다음과 같습니다.

```text
numpy
pandas
scikit-learn
joblib
```

가상환경은 필수는 아닙니다. 로컬에서 패키지 충돌을 피하고 싶을 때만 사용하면 됩니다.

---

## 3. 입력 데이터 위치

평가용 TSV 파일을 예를 들어 `data/test/` 폴더에 넣습니다.

```text
data/test/
├── LUAD_testset_rna_expression_tumor.tsv
├── LUAD_testset_rna_expression_nat.tsv
├── LUAD_testset_protein_expression_tumor.tsv
├── LUAD_testset_protein_expression_nat.tsv
├── LSCC_testset_rna_expression_tumor.tsv
├── LSCC_testset_rna_expression_nat.tsv
├── LSCC_testset_protein_expression_tumor.tsv
└── LSCC_testset_protein_expression_nat.tsv
```

파일명은 완전히 위와 같을 필요는 없지만, 파일명 안에 다음 정보가 들어 있어야 자동 인식됩니다.

```text
LUAD 또는 LSCC
rna 또는 protein
expression
tumor 또는 nat
.tsv
```

예를 들어 다음 파일명은 인식됩니다.

```text
LUAD_trainingset_rna_expression_tumor.tsv
LSCC_testset_protein_expression_nat.tsv
my_LUAD_rna_expression_tumor.tsv
```

---

## 4. 실행 방법

### Task 1: Tumor vs Normal

```bash
python src/predict_tumor_normal.py --test_dir data/test
```

출력 예시는 다음과 같습니다.

```text
=== Task 1: Tumor vs Normal Classification ===
test_dir: data/test
model_family: rna_protein
samples: 34

=== 예측 결과 ===
sample_id cancer_type prediction prob_tumor
C3N-01799        LUAD      Tumor     1.0000
C3N-01799        LUAD     Normal     0.0000
```

### Task 2: LUAD vs LSCC

```bash
python src/predict_luad_lscc.py --test_dir data/test
```

출력 예시는 다음과 같습니다.

```text
=== Task 2: LUAD vs LSCC Classification ===
test_dir: data/test
model_family: rna_protein
samples: 17

=== 예측 결과 ===
sample_id prediction prob_LUAD prob_LSCC
C3N-01799       LUAD    0.9981    0.0019
C3N-02575       LSCC    0.0024    0.9976
```

### Task 3: Survival Prediction

```bash
python src/predict_survival.py --test_dir data/test
```

출력 예시는 다음과 같습니다.

```text
=== Task 3: Survival Prediction ===
test_dir: data/test
model_family: rna_protein
samples: 17
decision_rule: Death if prob_death >= 0.9500; otherwise Survival

=== 예측 결과 ===
sample_id cancer_type prediction prob_death prob_survival
C3N-01799        LUAD   Survival     0.2401        0.7599
C3N-02575        LSCC      Death     0.9712        0.0288
```

---

## 5. 출력 행 수가 너무 많을 때

기본값은 모든 예측 결과를 터미널에 출력합니다.

앞부분 일부만 보고 싶으면 `--max_rows` 옵션을 사용합니다.

```bash
python src/predict_tumor_normal.py --test_dir data/test --max_rows 20
python src/predict_luad_lscc.py --test_dir data/test --max_rows 20
python src/predict_survival.py --test_dir data/test --max_rows 20
```

`--max_rows 0`은 전체 출력입니다.

---

## 6. 사용 모델

이 프로젝트는 공개 training TSV로 미리 학습한 scikit-learn 기반 모델을 사용합니다.

- 결측치 처리: training median imputation
- 표준화: training mean/std 기반 z-score scaling
- feature selection: training data에서 class 구분에 유리한 feature 선택
- classifier: regularized Logistic Regression 기반 ensemble
- 저장 형식: `joblib`

모델 파일은 다음 위치에 있습니다.

```text
models/lung_cancer_models.joblib
```

이 파일이 없으면 평가용 TSV만으로는 예측할 수 없습니다. GitHub 제출 시 반드시 포함해야 합니다.

---

## 7. RNA/protein 입력 처리

스크립트는 기본적으로 `--model_family auto`를 사용합니다.

- RNA와 protein 파일이 모두 있으면 `rna_protein` 모델 사용
- RNA만 있으면 `rna` 모델 사용
- protein만 있으면 `protein` 모델 사용

직접 지정하려면 다음처럼 실행할 수 있습니다.

```bash
python src/predict_tumor_normal.py --test_dir data/test --model_family rna
python src/predict_luad_lscc.py --test_dir data/test --model_family protein
python src/predict_survival.py --test_dir data/test --model_family rna_protein
```

---

## 8. 로컬 작동 테스트 방법

공개 training TSV를 임시 test 데이터처럼 사용해서 코드가 정상 작동하는지 확인할 수 있습니다.

먼저 공개 TSV 파일들을 다음 위치에 넣습니다.

```text
data/public_train/
├── LUAD_trainingset_rna_expression_tumor.tsv
├── LUAD_trainingset_rna_expression_nat.tsv
├── LUAD_trainingset_protein_expression_tumor.tsv
├── LUAD_trainingset_protein_expression_nat.tsv
├── LUAD_trainingset_overall_survival.tsv
├── LSCC_trainingset_rna_expression_tumor.tsv
├── LSCC_trainingset_rna_expression_nat.tsv
├── LSCC_trainingset_protein_expression_tumor.tsv
├── LSCC_trainingset_protein_expression_nat.tsv
└── LSCC_trainingset_overall_survival.tsv
```

그다음 세 스크립트를 실행합니다.

```bash
python src/predict_tumor_normal.py --test_dir data/public_train --max_rows 10
python src/predict_luad_lscc.py --test_dir data/public_train --max_rows 10
python src/predict_survival.py --test_dir data/public_train --max_rows 10
```

세 명령 모두 터미널에 표가 출력되면 정상적으로 작동하는 것입니다.

공개 training data 기준 sanity accuracy를 확인하려면 다음 명령을 실행합니다.

```bash
python scripts/check_public_training_accuracy.py --data_dir data/public_train
```

예상 출력 형식은 다음과 같습니다.

```text
=== Public Training Sanity Check ===
data_dir: data/public_train
model_family: rna_protein
Task 1 Tumor vs Normal accuracy: 1.0000
Task 2 LUAD vs LSCC accuracy:    1.0000
Task 3 Survival accuracy:        0.9074

Note: this is a sanity check on public training data, not hidden-test performance.
```

이 값은 공개 training data를 다시 예측한 결과이므로 hidden test 성능을 의미하지 않습니다.  
단지 모델 로드, TSV 파일 인식, feature 정렬, 예측 과정이 정상 작동하는지 확인하기 위한 sanity check입니다.

---

## 9. 모델 재학습

공개 training TSV를 사용해 모델을 다시 만들고 싶으면 다음 명령을 사용할 수 있습니다.

```bash
python src/train_model.py --train_dir data/public_train --model_path models/lung_cancer_models.joblib
```

재학습하면 기존 `models/lung_cancer_models.joblib`가 덮어써질 수 있습니다.

---

## 10. 제출 시 주의사항

GitHub에는 다음 파일과 폴더를 포함해야 합니다.

```text
README.md
requirements.txt
models/lung_cancer_models.joblib
models/lung_cancer_models.summary.json
src/common.py
src/predict_tumor_normal.py
src/predict_luad_lscc.py
src/predict_survival.py
src/train_model.py
scripts/check_public_training_accuracy.py
```

다음은 올릴 필요가 없습니다.

```text
.venv/
__pycache__/
data/
개인 테스트용 출력 결과
```
