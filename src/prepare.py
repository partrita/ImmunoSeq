import os
import glob
import json
import csv
import gzip
import pickle
from bloom_filter2 import BloomFilter
from utils import get_kmers

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
data_dir = os.path.join(project_root, "data")


def prepare_human():
    human_json_path = os.path.join(data_dir, "HUMAN.json")
    assert os.path.exists(human_json_path)
    with open(human_json_path, "r") as f:
        data = json.load(f)
    return data.keys()


def prepare_oas_mouse():
    seqs = []
    oas_mouse_dir = os.path.join(data_dir, "OAS_PAIRED_MOUSE")
    assert os.path.exists(oas_mouse_dir)
    for path in glob.glob(os.path.join(oas_mouse_dir, "*.csv.gz")):
        f = gzip.open(path, "rt")
        csv_reader = csv.reader(f)
        meta = next(csv_reader)
        meta = json.loads(meta[0])
        header = next(csv_reader)
        index_map = {
            key: i
            for i, key in enumerate(header)
            if key in ["sequence_alignment_aa_heavy", "sequence_alignment_aa_light"]
        }
        for line in csv_reader:
            for _, idx in index_map.items():
                seqs.append(line[idx])
    return seqs


def prepare_oas_human():
    seqs = []
    oas_human_dir = os.path.join(data_dir, "OAS_PAIRED_HUMAN")
    assert os.path.exists(oas_human_dir)
    for path in glob.glob(os.path.join(oas_human_dir, "*.csv.gz")):
        f = gzip.open(path, "rt")
        csv_reader = csv.reader(f)
        meta = next(csv_reader)
        meta = json.loads(meta[0])
        if meta["Species"] != "human":
            continue
        if meta["Disease"] != "None":
            continue
        header = next(csv_reader)
        index_map = {
            key: i
            for i, key in enumerate(header)
            if key in ["sequence_alignment_aa_heavy", "sequence_alignment_aa_light"]
        }
        for line in csv_reader:
            for _, idx in index_map.items():
                seqs.append(line[idx])
    return seqs


def dump_seqs(seqs, database):
    for mer in range(8, 13):
        datas = []
        for seq in seqs:
            kmer = get_kmers(seq, mer, mer)
            datas += kmer
        datas = list(set(datas))
        pkl = os.path.join(data_dir, f"{database}_{str(mer)}mer.dump")
        if not os.path.exists(pkl):
            pools = BloomFilter(max_elements=1.5e8)
            for item in datas:
                pools.add(item)
            with open(pkl, "wb") as f:
                pickle.dump(pools, f)


if __name__ == "__main__":
    human_protein_seqs = prepare_human()
    oas_paired_mouse_seqs = prepare_oas_mouse()
    oas_paired_human_seqs = prepare_oas_human()
    for seqs, database in zip(
        [human_protein_seqs, oas_paired_mouse_seqs, oas_paired_human_seqs],
        ["human", "oas_paired_mouse", "oas_paired_human"],
    ):
        dump_seqs(seqs, database)
