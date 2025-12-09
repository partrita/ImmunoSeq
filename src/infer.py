from utils import generate_single_mutation, get_kmers, check_response
import pandas as pd
import numpy as np
import pickle
import os
import sys
from args import conf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
data_dir = os.path.join(project_root, "data")
design_dir = os.path.join(project_root, "design")
os.makedirs(design_dir, exist_ok=True)


def load_kmer_pools(mer, pools_human, pools_oas_pair_human, pools_oas_pair_mouse):
    """
    특정 K-mer 길이에 대한 K-mer 풀 데이터를 로드합니다.
    """
    try:
        if mer not in pools_human:
            pkl = os.path.join(data_dir, f"human_{mer}mer.dump")
            with open(pkl, "rb") as f:
                pools_human[mer] = pickle.load(f)
        if mer not in pools_oas_pair_human:
            pkl = os.path.join(data_dir, f"oas_paired_human_{mer}mer.dump")
            with open(pkl, "rb") as f:
                pools_oas_pair_human[mer] = pickle.load(f)
        if mer not in pools_oas_pair_mouse:
            pkl = os.path.join(data_dir, f"oas_paired_mouse_{mer}mer.dump")
            with open(pkl, "rb") as f:
                pools_oas_pair_mouse[mer] = pickle.load(f)
    except FileNotFoundError as e:
        print(f"오류: K-mer 풀 파일이 없습니다: {e}")
        # K-mer 풀 파일이 없으면 즉시 종료
        sys.exit(1)


def infer_mut(seqs, name, fixed_VH=[], fixed_VL=[], topk=1):  # fixed = [27-31,32,65-68]
    # K-mer 풀 딕셔너리는 함수 호출마다 초기화되지 않도록 처리하거나,
    # 함수 외부에 정의된 캐시를 사용하는 것이 효율적이지만, 원본 구조를 따릅니다.
    pools_human, pools_oas_pair_human, pools_oas_pair_mouse = {}, {}, {}
    pred, mut, top_seq = [], [], []
    seqs = seqs.split(",")

    # 1. 단일 변이 시퀀스 생성
    single_mut_vh = generate_single_mutation(seqs[0], annotation="VH_", fixed=fixed_VH)
    single_mut_vl = generate_single_mutation(seqs[1], annotation="VL_", fixed=fixed_VL)

    # VH 변이 시퀀스에 VL WT 시퀀스를 붙입니다.
    for k, v in single_mut_vh.items():
        single_mut_vh[k] = f"{v},{seqs[1]}"
    # VL 변이 시퀀스에 VH WT 시퀀스를 붙입니다.
    for k, v in single_mut_vl.items():
        single_mut_vl[k] = f"{seqs[0]},{v}"

    # WT 시퀀스를 포함한 전체 변이 시퀀스 딕셔너리 생성
    single_mut = {**single_mut_vh, **single_mut_vl, "wt": f"{seqs[0]},{seqs[1]}"}

    # 2. 각 변이 시퀀스에 대한 점수 계산
    for k, va in single_mut.items():
        scores = {}
        sub_pred = []
        v = va.split(",")  # v[0]=VH, v[1]=VL

        # VH 및 VL 시퀀스에 대해 독립적으로 점수를 계산합니다.
        for s in v:
            response_s, total_kmers_s = 0, 0
            for mer in range(conf.min_mer, conf.max_mer):
                # K-mer 풀 로드 (필요 시)
                load_kmer_pools(
                    mer, pools_human, pools_oas_pair_human, pools_oas_pair_mouse
                )

                # K-mer 추출
                kmers = get_kmers(s, mer, mer)

                # K-mer 응답 확인 (check_response)
                pool_pos_mer = [pools_human[mer], pools_oas_pair_human[mer]]
                pool_neg_mer = [pools_oas_pair_mouse[mer]]
                subscores, subresponse = check_response(
                    kmers, mer, pool_pos_mer, pool_neg_mer, scores
                )

                # 결과 누적
                scores = subscores
                response_s += subresponse
                total_kmers_s += len(kmers)

            # 시퀀스(VH 또는 VL)별 평균 점수 저장 (0으로 나누는 것 방지)
            sub_pred.append(response_s / max(1, total_kmers_s))

        # VH와 VL 점수의 평균을 최종 예측 점수로 사용
        pred.append(np.mean(sub_pred))
        mut.append(k)
        top_seq.append(va)

    # 3. 결과 정리 및 출력
    result = {"mut": mut, "score": pred, "seq": top_seq}
    df = pd.DataFrame(result)
    df_sort = df.sort_values("score", ascending=False)

    if topk > 0:
        # 상위 K개 결과 반환
        topk_seq = df_sort.head(topk)
        return topk_seq.to_dict(orient="records")
    else:
        # 전체 결과 CSV 파일로 저장
        output_filepath = os.path.join(design_dir, f"{name}_infer_mut_oneshot.csv")
        df_sort.to_csv(output_filepath, index=None)


def predict_ada(seq, pool_pos=[], pool_neg=[]):
    # predict_ada 함수는 K-mer 풀 로딩 로직이 외부에서 전달되는 pool_pos/pool_neg에 의존합니다.
    # 이 함수 자체에는 큰 로직 오류는 없으나, K-mer 개수가 0인 경우를 방지합니다.
    scores, response, total_kmers = {}, 0, 0
    seq = seq.split(",")
    for s in seq:
        for mer in range(conf.min_mer, conf.max_mer):
            kmers = get_kmers(s, mer, mer)
            # pool_pos와 pool_neg는 {mer: pool_data} 형태의 딕셔너리 리스트여야 함
            pool_mer_pos = [item[mer] for item in pool_pos if mer in item]
            neg_pool_mer = [item[mer] for item in pool_neg if mer in item]

            subscores, subresponse = check_response(
                kmers, mer, pool_mer_pos, neg_pool_mer, scores
            )
            response += subresponse
            total_kmers += len(kmers)
            scores = subscores

    return response / max(1, total_kmers)  # 0으로 나누는 것 방지


if __name__ == "__main__":
    # Load sequences from the specified input file
    input_seq_filepath = os.path.join(project_root, conf.input_sequences_file)
    try:
        input_df = pd.read_csv(input_seq_filepath)
        seqs = {}
        for _, row in input_df.iterrows():
            name = row["name"]
            vh_seq = row["VH"]
            vl_seq = row["VL"]
            if pd.notna(vh_seq):
                seqs[f"VH_{name}"] = vh_seq
            if pd.notna(vl_seq):
                seqs[f"VL_{name}"] = vl_seq
    except FileNotFoundError:
        print(
            f"\n🚨 오류: 필수 시퀀스 파일 '{input_seq_filepath}'을(를) 찾을 수 없습니다."
        )
        print("파일 경로를 확인하거나, 'data/' 디렉터리에 해당 파일을 배치해야 합니다.")
        sys.exit(1)

    # Load fixed mutations file if provided
    exp_mut = {}
    if conf.fixed_mutations_file:
        mutations_filepath = os.path.join(project_root, conf.fixed_mutations_file)
        try:
            with open(mutations_filepath, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if i % 5 == 0:
                        exp_mut[lines[i].strip()] = {
                            "VH": lines[i + 1].strip(),
                            "VL": lines[i + 2].strip(),
                            "VH_cdr": lines[i + 3].strip(),
                            "VL_cdr": lines[i + 4].strip(),
                        }
        except FileNotFoundError:
            print(
                f"\n🚨 오류: 필수 변이 파일 '{mutations_filepath}'을(를) 찾을 수 없습니다."
            )
            print("파일 경로를 확인해야 합니다.")
            sys.exit(1)

    # 3. 데이터 준비 및 초기화
    vhvl_seq, fixed = {}, {}
    for name, seq in seqs.items():
        fname = name.split("_")[-1]  # 항체 이름 추출 (예: CD28)
        # `exp_mut`에 해당 `fname` 키가 없는 경우, 이 항목을 건너뜁니다.
        # 예를 들어, "nb9"와 같은 항목은 `humab_25_mutations.txt`에 정보가 없습니다.
        if fname not in exp_mut:
            continue

        chain = name.split("_")[0]  # 체인 추출 (VH 또는 VL)

        # exp_mut에서 CDR 영역 인덱스 추출 (FileNotFoundError 처리됨)
        fixed_region = exp_mut[fname][f"{chain}_cdr"].split(",")

        if name not in fixed.keys():
            fixed[name] = []
            for i in range(3):
                # CDR 영역을 '시작-끝' 형태로 저장
                fixed[name].append(f"{fixed_region[i * 2]}-{fixed_region[i * 2 + 1]}")

        if fname not in vhvl_seq.keys():
            vhvl_seq[fname] = {}
        vhvl_seq[fname][chain] = seq

    # 4. One-shot Infer 실행 (전체 결과 CSV 파일 저장)
    print("\n--- One-shot Infer 실행 (최적 변이 탐색) ---")
    for name in vhvl_seq.keys():
        vh, vl = vhvl_seq[name]["VH"], vhvl_seq[name]["VL"]
        # 'fixed' 딕셔너리에서 키 이름이 'VH_CD28'과 같이 정확히 일치해야 합니다.
        fixed_VH, fixed_VL = fixed[f"VH_{name}"], fixed[f"VL_{name}"]
        # topk=0 이므로 결과를 CSV 파일로 저장합니다.
        infer_mut(f"{vh},{vl}", name, fixed_VH=fixed_VH, fixed_VL=fixed_VL, topk=0)
        print(
            f"  {name}: 단일 변이 스코어 분석 완료 및 'design/{name}_infer_mut_oneshot.csv' 저장"
        )

    # 5. Iteration Infer 실행 (반복 변이 탐색 및 파일 기록)
    print("\n--- Iteration Infer 실행 ---")

    # conf.infer_round와 conf.top_rank가 args 모듈에 정의되어 있어야 합니다.
    # 해당 값들이 정의되지 않은 경우 NameError가 발생할 수 있지만, 원본 코드를 유지합니다.

    for name in vhvl_seq.keys():
        top_seqs = []
        vh, vl = vhvl_seq[name]["VH"], vhvl_seq[name]["VL"]
        fixed_VH, fixed_VL = fixed[f"VH_{name}"], fixed[f"VL_{name}"]

        # 반복 결과 저장 파일 초기화
        output_file = os.path.join(design_dir, f"{name}.txt")
        if os.path.exists(output_file):
            os.remove(output_file)
        print(f"\n[{name}] 반복 변이 탐색 시작:")

        for n in range(conf.infer_round):
            if n == 0:
                # 0번째 라운드: WT 시퀀스에서 시작하여 상위 K개 변이 탐색
                top_seqs = infer_mut(
                    f"{vh},{vl}",
                    name,
                    fixed_VH=fixed_VH,
                    fixed_VL=fixed_VL,
                    topk=conf.top_rank,
                )

                # 결과 기록
                print(
                    f"  라운드 {n}: {top_seqs[0]['mut']} (Score: {top_seqs[0]['score']:.4f})"
                )
                with open(output_file, "a") as fin:
                    fin.write(
                        f"{n},{name},{top_seqs[0]['mut']},{top_seqs[0]['score']:.4f},{top_seqs[0]['seq']}\n"
                    )

            elif n >= 1:
                new_top_seqs = []
                for (
                    item
                ) in top_seqs:  # 이전 라운드의 상위 K개 시퀀스를 기반으로 변이 탐색
                    vh_prev, vl_prev = item["seq"].split(",")

                    # 현재 시퀀스를 기반으로 단일 변이 탐색 (상위 K개)
                    top_seq_subset = infer_mut(
                        f"{vh_prev},{vl_prev}",
                        name,
                        fixed_VH=fixed_VH,
                        fixed_VL=fixed_VL,
                        topk=conf.top_rank,
                    )

                    # 변이 이력 업데이트
                    for item2 in top_seq_subset:
                        # item2['mut']에는 이 라운드에서 발생한 단일 변이만 포함되어 있습니다.
                        # 이전 변이 이력(item['mut'])과 결합합니다.
                        # 단, item['mut']가 'wt'인 경우 불필요한 ','를 피합니다.
                        if item["mut"] == "wt":
                            item2["mut"] = item2["mut"]
                        else:
                            item2["mut"] = item["mut"] + "," + item2["mut"]
                        new_top_seqs.append(item2)

                # 겹치는 시퀀스를 제거하고, 점수를 기준으로 상위 K개만 선택
                top_seqs = sorted(new_top_seqs, key=lambda x: x["score"], reverse=True)[
                    : conf.top_rank
                ]

                # 결과 기록
                print(
                    f"  라운드 {n}: {top_seqs[0]['mut']} (Score: {top_seqs[0]['score']:.4f})"
                )
                with open(output_file, "a") as fin:
                    fin.write(
                        f"{n},{name},{top_seqs[0]['mut']},{top_seqs[0]['score']:.4f},{top_seqs[0]['seq']}\n"
                    )
