import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# 어텐션 계산 함수 정의
def compute_attention(querys, keys, values, is_causal=False):
    dim_k = querys.size(-1)  # 마지막 차원의 크기 (d_k)
    # 스케일 조정된 점수 계산
    scores = querys @ keys.transpose(-2, -1) / sqrt(dim_k)
    if is_causal:
        query_length = querys.size(1)
        key_length = keys.size(1)
        # 상삼각형 마스크 생성 (미래 정보 차단)
        temp_mask = torch.ones(query_length, key_length, dtype=torch.bool, device=scores.device).tril()
        # 마스크 적용하여 미래 위치의 점수에 -무한대 할당
        scores = scores.masked_fill(~temp_mask, float("-inf"))
    # 소프트맥스 함수로 가중치 계산
    weights = F.softmax(scores, dim=-1)
    # 가중치와 값의 곱으로 최종 어텐션 출력 계산
    return weights @ values

# Pre-Layer Normalization 피드포워드 네트워크 정의
class PreLayerNormFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 첫 번째 선형 변환
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 두 번째 선형 변환
        self.dropout1 = nn.Dropout(dropout)  # 첫 번째 드롭아웃 레이어
        self.dropout2 = nn.Dropout(dropout)  # 두 번째 드롭아웃 레이어
        self.activation = nn.GELU()  # 활성화 함수
        self.norm = nn.LayerNorm(d_model)  # 레이어 정규화

    def forward(self, src):
        x = self.norm(src)  # 레이어 정규화 적용
        x = self.activation(self.linear1(x))  # 첫 번째 선형 변환 후 활성화 함수 적용
        x = self.dropout1(x)  # 드롭아웃 적용
        x = self.linear2(x)  # 두 번째 선형 변환
        x = self.dropout2(x)  # 드롭아웃 적용
        x = x + src  # 잔차 연결
        return x

# Transformer 인코더 레이어 정의
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)  # 첫 번째 레이어 정규화
        self.dropout1 = nn.Dropout(dropout)  # 첫 번째 드롭아웃 레이어
        self.feed_forward = PreLayerNormFeedForward(d_model, dim_feedforward, dropout)  # 피드포워드 네트워크

    def forward(self, src):
        # 어텐션 서브레이어
        norm_x = self.norm1(src)  # 레이어 정규화 적용
        attn_output = compute_attention(norm_x, norm_x, norm_x)  # 어텐션 계산
        x = src + self.dropout1(attn_output)  # 잔차 연결 및 드롭아웃 적용

        # 피드포워드 서브레이어
        x = self.feed_forward(x)  # 피드포워드 네트워크 통과
        return x

# 메인 함수 정의
def main():
    # 디바이스 설정 (GPU 사용 가능 시 CUDA 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 샘플 입력 텍스트 (한국어)
    input_text = "나는 최근 파리 여행을 다녀왔다"
    input_text_list = input_text.split()
    print("input_text_list:", input_text_list)

    # 토큰을 ID로, ID를 토큰으로 매핑하는 딕셔너리 생성
    str2idx = {word: idx for idx, word in enumerate(input_text_list)}
    idx2str = {idx: word for idx, word in enumerate(input_text_list)}
    print("str2idx:", str2idx)
    print("idx2str:", idx2str)

    # 토큰을 ID로 변환
    input_ids = [str2idx[word] for word in input_text_list]
    print("input_ids:", input_ids)

    # 임베딩 차원과 최대 위치 정의
    embedding_dim = 16
    max_position = 12

    # 임베딩 레이어 초기화
    embed_layer = nn.Embedding(len(str2idx), embedding_dim).to(device)
    position_embed_layer = nn.Embedding(max_position, embedding_dim).to(device)

    # 위치 ID 생성 및 위치 임베딩 획득
    position_ids = torch.arange(len(input_ids), dtype=torch.long, device=device).unsqueeze(0)
    position_encodings = position_embed_layer(position_ids)

    # 입력 ID를 텐서로 변환하고 토큰 임베딩 획득
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    token_embeddings = embed_layer(input_ids_tensor)

    # 토큰 임베딩과 위치 임베딩 결합
    input_embeddings = (token_embeddings + position_encodings.squeeze(0)).unsqueeze(0)
    print("input_embeddings.shape:", input_embeddings.shape)

    # TransformerEncoderLayer 초기화
    dim_feedforward = 64  # 피드포워드 네트워크의 차원
    dropout = 0.1
    transformer_encoder_layer = TransformerEncoderLayer(
        d_model=embedding_dim,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)

    # TransformerEncoderLayer를 통해 순전파 수행
    after_encoder = transformer_encoder_layer(input_embeddings)
    print("TransformerEncoderLayer output shape (after_encoder.shape):", after_encoder.shape)

    # TransformerEncoderLayer 출력 결과
    print("TransformerEncoderLayer Output:", after_encoder)

if __name__ == "__main__":
    main()
