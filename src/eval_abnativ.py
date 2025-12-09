import pickle
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# 사용자 정의 유틸리티 함수 임포트 (외부 파일 의존성 유지)
from utils import get_kmers, check_response, emit_metrics

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
data_dir = os.path.join(project_root, "data")

# --- 상수 정의 ---

# 분석할 K-mer 길이 범위 (최소 8개, 최대 12개, range(8, 13)이므로 13은 포함되지 않음)
MIN_MER, MAX_MER = 8, 13

# 테스트 데이터셋 파일 이름 정의
TEST_DATASETS = {
    "vh": {
        "human": "test_human_VH_shuf_no_dupli_aligned_10k.txt",
        "diverse": "diverse_5_human_VH_biophi_and_test_seqs.txt",
        "mouse": "mouse_VH_cleaned_aligned_10k.txt",
        "rhesus": "rhesus_VH_cleaned_aligned_10k.txt",
        "pssm": "rd_pssm_human_VH_random_10k.txt",
    },
    "vkappa": {
        "human": "test_human_VKappa_shuf_no_dupli_aligned_10k.txt",
        "diverse": "diverse_2_5_human_VKappa_biophi_and_test_seqs.txt",
        "mouse": "mouse_VKappa_cleaned_aligned_10k.txt",
        "rhesus": "rhesus_VKappa_cleaned_aligned_10k.txt",
        "pssm": "rd_pssm_human_VKappa_random_10k.txt",
    },
    "vlambda": {
        "human": "test_human_VLambda_shuf_no_dupli_aligned_10k.txt",
        "diverse": "diverse_2_5_VLambda_biophi_and_test_seqs.txt",
        "mouse": "mouse_VLambda_cleaned_aligned_all.txt",
        "rhesus": "rhesus_VLambda_cleaned_aligned_10k.txt",
        "pssm": "rd_pssm_human_VLambda_random_10k.txt",
    },
}

# --- 헬퍼 함수 ---


def read_txt(filepath: str) -> list:
    """
    지정된 경로의 텍스트 파일에서 시퀀스를 읽어와 리스트로 반환합니다.
    '-' 문자는 제거하고 각 줄의 공백을 제거합니다.

    Args:
        filepath: 읽어올 파일의 전체 경로.

    Returns:
        정제된 시퀀스 문자열 리스트. 파일이 없으면 None 반환.
    """
    if not os.path.exists(filepath):
        # 파일이 존재하지 않으면 None을 반환하며 함수를 종료합니다.
        return None

    sequences = []
    with open(filepath, "r") as f:
        lines = f.readlines()
        # 각 줄에서 공백을 제거하고 '-' 문자를 제거한 후 리스트에 추가합니다.
        for line in lines:
            sequences.append(line.strip().replace("-", ""))
    return sequences


def load_kmer_pools(min_mer: int, max_mer: int) -> tuple:
    """
    지정된 K-mer 범위에 대해 Pickle 파일에서 K-mer 풀 데이터를 로드합니다.

    Args:
        min_mer: 최소 K-mer 길이.
        max_mer: 최대 K-mer 길이 (이 값 미만까지 로드).

    Returns:
        로딩된 K-mer 풀 딕셔너리 3개 (human, paired_human, paired_mouse)를 튜플로 반환합니다.
        각 딕셔너리는 {mer_length: pool_data} 형태입니다.
    """
    kmer_pools_human = {}
    kmer_pools_paired_human = {}
    kmer_pools_paired_mouse = {}

    for mer in range(min_mer, max_mer):
        # Human K-mer 풀 로드
        pkl_human = os.path.join(data_dir, f"human_{mer}mer.dump")
        with open(pkl_human, "rb") as f:
            kmer_pools_human[mer] = pickle.load(f)

        # Paired Human K-mer 풀 로드
        pkl_paired_human = os.path.join(data_dir, f"oas_paired_human_{mer}mer.dump")
        with open(pkl_paired_human, "rb") as f:
            kmer_pools_paired_human[mer] = pickle.load(f)

        # Paired Mouse K-mer 풀 로드
        pkl_paired_mouse = os.path.join(data_dir, f"oas_paired_mouse_{mer}mer.dump")
        with open(pkl_paired_mouse, "rb") as f:
            kmer_pools_paired_mouse[mer] = pickle.load(f)

    return kmer_pools_human, kmer_pools_paired_human, kmer_pools_paired_mouse


def calc_scores(sequences: list, pool_pos: list, pool_neg: list, label: int) -> tuple:
    """
    주어진 시퀀스 리스트에 대해 K-mer 풀을 사용하여 점수를 계산합니다.

    Args:
        sequences: 점수를 계산할 시퀀스 문자열 리스트.
        pool_pos: 양성(Positive) K-mer 풀 딕셔너리 리스트.
        pool_neg: 음성(Negative) K-mer 풀 딕셔너리 리스트.
        label: 이 시퀀스들의 참 레이블 (1 또는 0).

    Returns:
        (예측 점수 리스트, 참 레이블 리스트)
    """
    # 입력 시퀀스 리스트는 비어 있으면 안 됩니다.
    assert len(sequences) > 0, "입력 시퀀스 리스트가 비어 있습니다."

    pred_scores, true_labels = [], []

    # 각 시퀀스에 대해 점수를 계산합니다.
    for sequence in sequences:
        # 시퀀스당 초기화
        scores_by_mer, total_response, total_kmers_count = {}, 0, 0

        # 지정된 K-mer 길이 범위에 대해 반복합니다.
        for mer in range(MIN_MER, MAX_MER):
            # 현재 K-mer 길이의 K-mer들을 추출합니다.
            kmers = get_kmers(sequence, mer, mer)

            # 현재 K-mer 길이에 해당하는 양성 풀 데이터를 준비합니다.
            pool_mer_pos = [item[mer] for item in pool_pos if mer in item]

            # 현재 K-mer 길이에 해당하는 음성 풀 데이터를 준비합니다.
            pool_mer_neg = [item[mer] for item in pool_neg if mer in item]

            # K-mer를 사용하여 응답 점수를 확인합니다. (utils.check_response 함수 사용)
            subscores, subresponse = check_response(
                kmers, mer, pool_mer_pos, pool_mer_neg, scores_by_mer
            )

            total_response += subresponse
            total_kmers_count += len(kmers)
            scores_by_mer = subscores

        # 시퀀스의 최종 예측 점수: (총 응답 / 총 K-mer 개수)
        # K-mer 개수가 0인 경우를 방지하기 위해 max(1, ...) 사용
        final_score = total_response / max(1, total_kmers_count)
        pred_scores.append(final_score)
        true_labels.append(label)

    return pred_scores, true_labels


def print_metrics(pred: list, true: list, nfloat: int = 5):
    """
    예측 점수와 참 레이블을 기반으로 다양한 분류 지표를 계산하고 출력합니다.

    Args:
        pred: 예측 점수 리스트 (0과 1 사이의 실수).
        true: 참 레이블 리스트 (0 또는 1).
        nfloat: 출력할 소수점 자리수.
    """
    # 0.0001 간격의 임계값(interval)을 생성합니다.
    intervals = np.arange(0, 1, 0.0001)

    # 임계값과 무관한 지표를 먼저 계산합니다.
    auroc = roc_auc_score(true, pred)
    auprc = average_precision_score(true, pred)

    # 출력 헤더를 인쇄합니다.
    print("interval,mcc,auprc,auroc,accuracy,f1_score,recall,precision")

    # 각 임계값에 대해 지표를 계산하고 출력합니다.
    for interval in intervals:
        # 임계값을 적용하여 예측값을 이진화합니다.
        fpred = [1 if score >= interval else 0 for score in pred]

        # 유틸리티 함수를 사용하여 정확도, F1 점수 등을 계산합니다.
        # emit_metrics 순서: (accuracy, f1_score, mcc, recall, precision)
        metric = emit_metrics(fpred, true)

        # 결과를 지정된 소수점 자리수로 반올림합니다.
        metric_rounded = list(map(lambda x: round(x, nfloat), metric))

        # 순서에 맞게 변수에 할당합니다.
        accuracy, f1_score, mcc, recall, precision = metric_rounded

        # 결과를 CSV 형식으로 출력합니다.
        print(
            f"{round(interval, nfloat)},{mcc},{auprc},{auroc},{accuracy},{f1_score},{recall},{precision}"
        )


# --- 메인 실행 로직 ---


def process_datasets(abnativ_dataset: str, pool_pos: list, pool_neg: list):
    """
    특정 데이터셋 타입에 대해 시퀀스를 로드하고 점수를 계산하며 결과를 출력합니다.
    """
    # 1. 데이터셋 파일 경로 설정 및 시퀀스 로드
    base_path = os.path.join(data_dir, "abnativ", abnativ_dataset)
    dataset_files = TEST_DATASETS[abnativ_dataset]

    # 각 데이터셋의 시퀀스를 로드합니다.
    diverse_seqs = read_txt(os.path.join(base_path, dataset_files["diverse"]))
    mouse_seqs = read_txt(os.path.join(base_path, dataset_files["mouse"]))
    rhesus_seqs = read_txt(os.path.join(base_path, dataset_files["rhesus"]))
    pssm_seqs = read_txt(os.path.join(base_path, dataset_files["pssm"]))
    human_seqs = read_txt(os.path.join(base_path, dataset_files["human"]))

    # 2. 각 시퀀스에 대해 점수 계산 (label=1은 양성, label=0은 음성)

    # 양성 데이터 (Target Label: 1)
    diverse_pred, diverse_true = calc_scores(
        diverse_seqs, pool_pos=pool_pos, pool_neg=pool_neg, label=1
    )
    human_pred, human_true = calc_scores(
        human_seqs, pool_pos=pool_pos, pool_neg=pool_neg, label=1
    )

    # 음성 데이터 (Target Label: 0)
    mouse_pred, mouse_true = calc_scores(
        mouse_seqs, pool_pos=pool_pos, pool_neg=pool_neg, label=0
    )
    rhesus_pred, rhesus_true = calc_scores(
        rhesus_seqs, pool_pos=pool_pos, pool_neg=pool_neg, label=0
    )
    pssm_pred, pssm_true = calc_scores(
        pssm_seqs, pool_pos=pool_pos, pool_neg=pool_neg, label=0
    )

    # 3. 다양한 이진 분류 태스크를 조합합니다.
    tasks = {
        "human_mouse": {
            "pred": human_pred + mouse_pred,
            "true": human_true + mouse_true,
        },
        "human_pssm": {"pred": human_pred + pssm_pred, "true": human_true + pssm_true},
        "human_rhesus": {
            "pred": human_pred + rhesus_pred,
            "true": human_true + rhesus_true,
        },
        "diverse_human_mouse": {
            "pred": diverse_pred + mouse_pred,
            "true": diverse_true + mouse_true,
        },
        "diverse_human_pssm": {
            "pred": diverse_pred + pssm_pred,
            "true": diverse_true + pssm_true,
        },
        "diverse_human_rhesus": {
            "pred": diverse_pred + rhesus_pred,
            "true": diverse_true + rhesus_true,
        },
    }

    # 4. 각 태스크의 분류 지표를 출력합니다.
    for task_name, v in tasks.items():
        print(f"--- {abnativ_dataset} :: {task_name} 분석 결과 ---")
        print_metrics(v["pred"], v["true"])
        print("-----------------------------------")


if __name__ == "__main__":
    # 1. K-mer 풀 데이터 로드 (모든 K-mer 길이에 대한 데이터)
    kmer_pools_human, kmer_pools_paired_human, kmer_pools_paired_mouse = (
        load_kmer_pools(MIN_MER, MAX_MER)
    )

    # 2. 양성 및 음성 풀 설정
    # 양성(Positive) 풀: 일반 Human 데이터 + Paired Human 데이터
    POOL_POS = [kmer_pools_human, kmer_pools_paired_human]
    # 음성(Negative) 풀: Paired Mouse 데이터
    POOL_NEG = [kmer_pools_paired_mouse]

    # 3. 분석할 데이터셋 타입 선택 및 실행 (예: 'vkappa' 선택)
    ABNATIV_DATASET = "vkappa"  # vh, vlambda, vkappa 중 선택

    process_datasets(ABNATIV_DATASET, POOL_POS, POOL_NEG)
