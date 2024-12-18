from transformers import AutoTokenizer

# 모델 ID 설정
model_id = 'klue/roberta-base'

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 개별 문장 리스트
individual_sentences = ['첫 번째 문장', '두 번째 문장']

# 결합된 문장 리스트 (리스트 안에 리스트)
combined_sentences = [['첫 번째 문장', '두 번째 문장']]

def tokenize_sentences(sentences, is_combined=False):
    """
    문장을 토큰화하는 함수.

    Args:
        sentences (list): 토큰화할 문장들의 리스트.
        is_combined (bool): 결합된 문장인지 여부.

    Returns:
        dict: 토큰화된 결과.
    """
    return tokenizer(sentences, is_split_into_words=is_combined, padding=True, truncation=True)

def decode_token_ids(token_ids):
    """
    토큰 ID를 디코딩하는 함수.

    Args:
        token_ids (list): 디코딩할 토큰 ID의 리스트.

    Returns:
        list: 디코딩된 문장들.
    """
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)

# 개별 문장 토큰화
tokenized_individual_sentences = tokenize_sentences(individual_sentences)
print("개별 문장 토큰화 결과:")
print(tokenized_individual_sentences)

# 개별 문장의 input_ids 추출
individual_input_ids = tokenized_individual_sentences['input_ids']

# 토큰 ID 디코딩
decoded_individual_sentences = decode_token_ids(individual_input_ids)
print("\n개별 문장 디코딩 결과:")
print(decoded_individual_sentences)
# ['첫 번째 문장', '두 번째 문장']

# 결합된 문장 토큰화
tokenized_combined_sentences = tokenize_sentences(combined_sentences, is_combined=True)
print("\n결합된 문장 토큰화 결과:")
print(tokenized_combined_sentences)

# 결합된 문장의 input_ids 추출
combined_input_ids = tokenized_combined_sentences['input_ids']

# 토큰 ID 디코딩
decoded_combined_sentence = decode_token_ids(combined_input_ids)
print("\n결합된 문장 디코딩 결과:")
print(decoded_combined_sentence)
# ['첫 번째 문장 두 번째 문장']
