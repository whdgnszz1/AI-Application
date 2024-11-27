import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

def compute_attention(querys, keys, values, is_causal=False):
    """
    어텐션 출력을 계산합니다.

    Args:
        querys (Tensor): 쿼리 텐서, 형태 (배치 크기, 헤드 수, 시퀀스 길이, 헤드 차원)
        keys (Tensor): 키 텐서, 형태 (배치 크기, 헤드 수, 시퀀스 길이, 헤드 차원)
        values (Tensor): 값 텐서, 형태 (배치 크기, 헤드 수, 시퀀스 길이, 헤드 차원)
        is_causal (bool): True인 경우, 미래 토큰에 대한 어텐션을 방지하는 인과 마스킹을 적용합니다.

    Returns:
        Tensor: 어텐션 출력, 형태 (배치 크기, 헤드 수, 시퀀스 길이, 헤드 차원)
    """
    dim_k = querys.size(-1)
    # 어텐션 점수 계산: (배치 크기, 헤드 수, 시퀀스 길이, 시퀀스 길이)
    scores = torch.matmul(querys, keys.transpose(-2, -1)) / sqrt(dim_k)

    if is_causal:
        # 미래 토큰에 대한 어텐션을 방지하는 마스크를 생성
        seq_length = scores.size(-1)
        mask = torch.tril(torch.ones((seq_length, seq_length), device=scores.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 소프트맥스를 통해 가중치 계산
    weights = F.softmax(scores, dim=-1)  # (배치 크기, 헤드 수, 시퀀스 길이, 시퀀스 길이)
    # 가중치를 값 텐서에 곱하여 어텐션 출력 계산
    return torch.matmul(weights, values)  # (배치 크기, 헤드 수, 시퀀스 길이, 헤드 차원)

class MultiheadAttention(nn.Module):
    def __init__(self, token_embed_dim, d_model, n_head, is_causal=False):
        """
        MultiheadAttention을 초기화합니다.

        Args:
            token_embed_dim (int): 토큰 임베딩의 차원
            d_model (int): 모델의 차원 (보통 token_embed_dim과 동일하게 설정)
            n_head (int): 어텐션 헤드의 수
            is_causal (bool): True인 경우, 인과 마스킹을 적용
        """
        super(MultiheadAttention, self).__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.is_causal = is_causal
        self.head_dim = d_model // n_head

        # 쿼리, 키, 값 벡터 생성을 위한 선형 층
        self.weight_q = nn.Linear(token_embed_dim, d_model, bias=False)
        self.weight_k = nn.Linear(token_embed_dim, d_model, bias=False)
        self.weight_v = nn.Linear(token_embed_dim, d_model, bias=False)

        # 여러 헤드의 출력을 합치기 위한 선형 층
        self.concat_linear = nn.Linear(d_model, d_model)

    def forward(self, querys, keys, values):
        """
        멀티헤드 어텐션의 순전파를 수행

        Args:
            querys (Tensor): 쿼리 텐서, 형태 (배치 크기, 시퀀스 길이, 토큰 임베딩 차원)
            keys (Tensor): 키 텐서, 형태 (배치 크기, 시퀀스 길이, 토큰 임베딩 차원)
            values (Tensor): 값 텐서, 형태 (배치 크기, 시퀀스 길이, 토큰 임베딩 차원)

        Returns:
            Tensor: 어텐션 출력, 형태 (배치 크기, 시퀀스 길이, d_model)
        """
        B, T, C = querys.size()

        # 쿼리, 키, 값 벡터 생성 후 헤드 수에 맞게 분할
        querys = self.weight_q(querys).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        keys = self.weight_k(keys).view(B, T, self.n_head, self.head_dim).transpose(1, 2)        # (B, n_head, T, head_dim)
        values = self.weight_v(values).view(B, T, self.n_head, self.head_dim).transpose(1, 2)    # (B, n_head, T, head_dim)

        # 어텐션 계산
        attention = compute_attention(querys, keys, values, self.is_causal)  # (B, n_head, T, head_dim)

        # 헤드들을 다시 결합
        output = attention.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        output = self.concat_linear(output)  # (B, T, d_model)

        return output
