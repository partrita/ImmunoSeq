import torch
import pickle
import pandas as pd
import torchmetrics.functional as metrics
import os
import sys

# 외부 파일에서 필요한 모듈 임포트
# 'utils' 모듈: K-mer 추출, 응답 확인, 플로팅 함수 포함
from utils import get_kmers, check_response, plot_ada

# 'args' 모듈: 설정(configuration) 객체 포함 (예: conf.min_mer, conf.max_mer)
from args import conf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
data_dir = os.path.join(project_root, "data")

# --- K-mer 풀 로딩 함수 ---


def load_kmer_pools(config) -> tuple:
    """
    설정된 K-mer 길이 범위(config.min_mer ~ config.max_mer)에 대해
    Pickle 파일에서 K-mer 풀 데이터를 로드합니다.

    Args:
        config: 최소/최대 K-mer 길이를 포함하는 설정 객체 (conf).

    Returns:
        로딩된 K-mer 풀 딕셔너리 3개 (human, paired_human, paired_mouse)를 튜플로 반환합니다.
        각 딕셔너리는 {mer_length: pool_data} 형태입니다.
    """
    # K-mer 풀 딕셔너리 초기화
    kmer_pools_human, kmer_pools_paired_human, kmer_pools_paired_mouse = {}, {}, {}

    # 설정된 범위의 K-mer 길이에 대해 반복합니다.
    for mer in range(config.min_mer, config.max_mer):
        try:
            # 일반 Human K-mer 풀 로드
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

        except FileNotFoundError as e:
            print(f"오류: K-mer 풀 파일이 없습니다: {e}")
            # 파일이 없으면 프로그램 종료
            sys.exit(1)

    return kmer_pools_human, kmer_pools_paired_human, kmer_pools_paired_mouse


# --- 벤치마크 함수 ---


def benckmark_ada(
    config,
    pool_pos: list,
    pool_neg: list,
    origin=["Human", "Humanized", "Humanized/Chimeric", "Chimeric", "Mouse"],
    plot=False,
) -> list:
    """
    항체의 면역원성(ADA) 데이터셋에 대해 K-mer 기반 예측 성능을 평가합니다.

    Args:
        config: 설정 객체 (K-mer 길이 범위 포함).
        pool_pos: 양성 K-mer 풀 리스트.
        pool_neg: 음성 K-mer 풀 리스트.
        origin: 분석에 포함할 항체의 종(Species) 리스트 (기본값 설정).
        plot: 개별 항체에 대한 점수 플롯 생성 여부 (기본값 설정).

    Returns:
        개별 항체별 K-mer 점수 계산 결과 리스트.
    """
    path = os.path.join(data_dir, "ADA_Clinical_Ab_2021.csv")
    try:
        # 1. 데이터 로드 및 필터링
        df = pd.read_csv(path, index_col=None)
    except FileNotFoundError:
        print(f"오류: 데이터셋 파일 {path}를 찾을 수 없습니다.")
        return []

    # 지정된 종(Species)의 항체만 필터링합니다.
    df = df[df["Species"].isin(origin)]

    # 필요한 데이터를 리스트로 추출합니다.
    names = df.Antibody.tolist()
    seqs_VH = df.VH.tolist()
    seqs_VL = df.VL.tolist()
    ada = df.Immunogenicity.tolist()  # 참 면역원성 값 (Label)
    specy = df.Species.tolist()

    # 예측 및 결과 저장을 위한 리스트 초기화
    pred, true = [], []
    report = []  # 개별 항체별 상세 계산 결과

    # 2. 각 항체에 대해 K-mer 점수 계산
    for i in range(len(names)):
        scores_VH, scores_VL = {}, {}
        response_VH, response_VL, total_kmers_VH, total_kmers_VL = 0, 0, 0, 0

        # 설정된 K-mer 길이 범위에 대해 반복합니다.
        for mer in range(config.min_mer, config.max_mer):
            # VH 및 VL 시퀀스에서 K-mer를 추출합니다.
            kmers_VH = get_kmers(seqs_VH[i], mer, mer)
            kmers_VL = get_kmers(seqs_VL[i], mer, mer)

            # 현재 K-mer 길이에 해당하는 양성 및 음성 풀 데이터를 준비합니다.
            pos_pool_mer = [item[mer] for item in pool_pos if mer in item]
            neg_pool_mer = [item[mer] for item in pool_neg if mer in item]

            # VH 및 VL에 대한 K-mer 응답 점수를 계산합니다.
            subscores_VH, subresponse_VH = check_response(
                kmers_VH, mer, pos_pool_mer, neg_pool_mer, scores_VH
            )
            subscores_VL, subresponse_VL = check_response(
                kmers_VL, mer, pos_pool_mer, neg_pool_mer, scores_VL
            )

            # 누적 응답 및 K-mer 개수를 업데이트합니다.
            response_VH += subresponse_VH
            response_VL += subresponse_VL
            total_kmers_VH += len(kmers_VH)
            total_kmers_VL += len(kmers_VL)
            scores_VH = subscores_VH
            scores_VL = subscores_VL

        # K-mer 개수가 0인 경우를 방지하여 점수를 계산합니다.
        score_VH = response_VH / max(1, total_kmers_VH)
        score_VL = response_VL / max(1, total_kmers_VL)

        # 최종 예측 점수: VH와 VL 점수의 평균
        final_prediction = (score_VH + score_VL) / 2

        # 3. 결과 저장
        if plot:
            # plot이 True인 경우 개별 플롯을 생성합니다.
            plot_ada(scores_VH, scores_VL, names[i], specy[i].replace("/", "_"))

        pred.append(final_prediction)
        true.append(ada[i])  # 참 레이블 저장
        # 상세 결과: (항체 이름, VH 응답, VL 응답, 총 VH K-mer, 총 VL K-mer)
        report.append(
            {
                "Antibody": names[i],
                "VH_Response": response_VH,
                "VL_Response": response_VL,
                "Total_Kmers_VH": total_kmers_VH,
                "Total_Kmers_VL": total_kmers_VL,
                "Prediction": final_prediction,
                "True_Label": ada[i],
            }
        )

    # 4. 성능 지표 계산
    pred_tensor = torch.tensor(pred, dtype=torch.float)
    true_tensor = torch.tensor(true, dtype=torch.float)

    # 피어슨 상관계수 (Pearson Correlation Coefficient)
    pearson = metrics.pearson_corrcoef(pred_tensor, true_tensor).item()
    # 스피어만 상관계수 (Spearman Correlation Coefficient)
    spearman = metrics.spearman_corrcoef(pred_tensor, true_tensor).item()

    # 5. 결과 출력 명확화
    print("\n============================================")
    print(f"✅ 면역원성 예측 성능 (총 항체 수 N={len(names)})")
    print(f"  - 피어슨 상관계수 (Pearson r): {pearson:.4f} (선형 관계 강도)")
    print(f"  - 스피어만 상관계수 (Spearman ρ): {spearman:.4f} (순위 일치 정도)")
    print("============================================")

    return report


# --- 메인 실행 함수 ---


def main():
    # 1. K-mer 풀 데이터 로드
    kmer_pools_human, kmer_pools_paired_human, kmer_pools_paired_mouse = (
        load_kmer_pools(conf)
    )

    # 2. 양성 및 음성 풀 설정
    # 양성(Positive) 풀: 일반 Human 데이터 + Paired Human 데이터
    pool_pos = [kmer_pools_human, kmer_pools_paired_human]
    # 음성(Negative) 풀: Paired Mouse 데이터
    pool_neg = [kmer_pools_paired_mouse]

    # 3. 벤치마크 실행 (누락된 origin과 plot 인자를 명시적으로 전달하여 TypeError 해결)
    # origin과 plot은 benckmark_ada 함수 정의에 설정된 기본값을 사용합니다.
    default_origin = ["Human", "Humanized", "Humanized/Chimeric", "Chimeric", "Mouse"]
    default_plot = False

    report = benckmark_ada(
        conf,
        pool_pos=pool_pos,
        pool_neg=pool_neg,
        origin=default_origin,
        plot=default_plot,
    )

    # 4. 상세 결과 출력 (첫 5개 항체만 예시로 출력)
    if report:
        print("\n=== 개별 항체별 상세 K-mer 점수 보고서 (상위 5개) ===")
        report_df = pd.DataFrame(report)
        print(
            report_df[["Antibody", "Prediction", "True_Label", "Total_Kmers_VH"]].head()
        )
        print(
            f"\n총 {len(report)}개 항체의 상세 결과가 'report' 변수에 저장되었습니다."
        )

    return report


if __name__ == "__main__":
    # 'conf' 객체가 외부 'args' 모듈에서 올바르게 로드되었는지 확인한 후 실행해야 합니다.
    main()
