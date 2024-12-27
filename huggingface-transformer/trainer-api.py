from datasets import load_dataset
import torch
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

# 1) 모델 및 토크나이저 설정
model_id = 'klue/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2) KLUE YNAT 데이터셋 로드
klue_tc_train = load_dataset('klue', 'ynat', split='train')
klue_tc_eval = load_dataset('klue', 'ynat', split='validation')

# 3) 필요없는 열 제거
klue_tc_train = klue_tc_train.remove_columns(['guid', 'url', 'date'])
klue_tc_eval = klue_tc_eval.remove_columns(['guid', 'url', 'date'])

# 4) 레이블 정보 확인 (0~6까지의 뉴스 카테고리)
klue_tc_label = klue_tc_train.features['label']

# 5) 레이블(정수) -> 문자열 변환: 디버깅/확인용
def make_str_label(batch):
    batch['label_str'] = klue_tc_label.int2str(batch['label'])
    return batch

klue_tc_train = klue_tc_train.map(make_str_label, batched=True, batch_size=1000)

# 6) 임의로 train/eval/test용으로 데이터 분할
#    - 실제로는 데이터 규모에 맞춰 test_size를 적절히 조정
train_dataset = klue_tc_train.train_test_split(test_size=1000, shuffle=True, seed=42)['test']
dataset = klue_tc_eval.train_test_split(test_size=1000, shuffle=True, seed=42)
test_dataset = dataset['test']
valid_dataset = dataset['train'].train_test_split(test_size=500, shuffle=True, seed=42)['test']

# 7) 입력 데이터(뉴스 제목)를 토큰화하는 함수
def tokenize_function(examples):
    return tokenizer(examples["title"], padding="max_length", truncation=True)

# 8) 모델 생성
#    - label 개수를 지정해야 하기 때문에 dataset.features['label'].num_classes 등을 사용
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=klue_tc_train.features['label'].num_classes
)

# 9) 실제 토큰화 적용
train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 10) 훈련 인자 설정
training_args = TrainingArguments(
    output_dir='./results',                 # 결과물이 저장될 디렉토리
    num_train_epochs=1,                     # 에폭 수
    per_device_train_batch_size=8,          # 훈련 배치 크기
    per_device_eval_batch_size=8,           # 평가 배치 크기
    eval_strategy='epoch',                  # 평가 전략: 매 에폭마다 평가 (변경됨)
    learning_rate=5e-5,                     # 학습률
    push_to_hub=False                       # 모델 Hub 업로드 여부
)

# 11) 메트릭 계산 함수 정의
def compute_metrics(eval_pred):
    # eval_pred는 (로짓, 정답 레이블)의 튜플
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # 로짓에서 가장 높은 스코어의 인덱스 추출
    return {"accuracy": (predictions == labels).mean()}

# 12) Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,     # 학습에 사용할 데이터셋
    eval_dataset=valid_dataset,      # 검증에 사용할 데이터셋
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # 메트릭 계산 함수
)

# 13) 학습 실행
trainer.train()

# 14) 테스트셋으로 최종 평가
print(trainer.evaluate(test_dataset))
