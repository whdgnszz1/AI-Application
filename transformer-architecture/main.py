import torch
import torch.nn as nn
from modules import MultiheadAttention

def main():
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 샘플 입력 텍스트 (한국어)
    input_text = "나는 최근 파리 여행을 다녀왔다"
    input_text_list = input_text.split()
    print("input_text_list:", input_text_list)

    # 토큰을 ID로, ID를 토큰으로 매핑하는 딕셔너리를 생성
    str2idx = {word: idx for idx, word in enumerate(input_text_list)}
    idx2str = {idx: word for idx, word in enumerate(input_text_list)}
    print("str2idx:", str2idx)
    print("idx2str:", idx2str)

    # 토큰을 ID로 변환
    input_ids = [str2idx[word] for word in input_text_list]
    print("input_ids:", input_ids)

    # 임베딩 차원과 최대 위치를 정의
    embedding_dim = 16
    max_position = 12

    # 임베딩 층을 초기화
    embed_layer = nn.Embedding(len(str2idx), embedding_dim).to(device)
    position_embed_layer = nn.Embedding(max_position, embedding_dim).to(device)

    # 위치 ID를 생성하고 해당 위치 임베딩을 가져온다.
    position_ids = torch.arange(len(input_ids), dtype=torch.long, device=device).unsqueeze(0)  # 형태: (1, 시퀀스 길이)
    position_encodings = position_embed_layer(position_ids)  # 형태: (1, 시퀀스 길이, 임베딩 차원)

    # 입력 ID를 텐서로 변환하고 토큰 임베딩을 가져온다.
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    token_embeddings = embed_layer(input_ids_tensor)  # 형태: (시퀀스 길이, 임베딩 차원)

    # 토큰 임베딩과 위치 임베딩을 결합한다.
    input_embeddings = (token_embeddings + position_encodings.squeeze(0)).unsqueeze(0)  # 형태: (1, 시퀀스 길이, 임베딩 차원)
    print("input_embeddings.shape:", input_embeddings.shape)

    # 멀티헤드 어텐션 초기화
    n_head = 4
    multihead_attention = MultiheadAttention(
        token_embed_dim=embedding_dim,
        d_model=embedding_dim,
        n_head=n_head,
        is_causal=False
    ).to(device)

    # 멀티헤드 어텐션을 통해 순전파를 수행
    after_attention_embeddings = multihead_attention(input_embeddings, input_embeddings, input_embeddings)
    print("어텐션 적용 후 형태 (after_attention_embeddings.shape):", after_attention_embeddings.shape)

    # 어텐션 출력
    print("Attention Output:", after_attention_embeddings)

if __name__ == "__main__":
    main()