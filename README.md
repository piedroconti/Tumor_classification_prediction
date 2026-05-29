# Lung Cancer Prediction Project

이 프로젝트는 폐암 데이터에 대해 다음 3가지 예측 문제를 수행합니다.

1. **Task 1: Tumor vs Normal Classification**  
   Tumor/NAT expression 데이터를 보고 `Tumor` 또는 `Normal`을 예측합니다.

2. **Task 2: LUAD vs LSCC Classification**  
   Tumor expression 데이터를 보고 `LUAD` 또는 `LSCC`를 예측합니다.

3. **Task 3: Survival Prediction**  
   Tumor expression 데이터를 보고 `Survival` 또는 `Death`를 예측합니다.



```text
src/predict_tumor_normal.py   # Task 1
src/predict_luad_lscc.py      # Task 2
src/predict_survival.py       # Task 3
```



## 1. 프로젝트 구조

```text
lung_cancer_prediction_taskwise_terminal/
├── README.md
├── requirements.txt
├── data/
│   └──test/         # 여기에 평가용 데이터 배치
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
따라서 훈련용 TSV 없이 평가용 TSV만 넣고 예측을 실행할 수 있습니다.

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


---

## 3. 입력 데이터 위치

평가용 TSV 파일을 `data/test/` 폴더에 넣습니다.

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
