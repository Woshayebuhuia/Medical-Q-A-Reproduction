import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/medical_new_2.json"
INDEX_DIR = "vector_store"
INDEX_PATH = os.path.join(INDEX_DIR, "medical.index")
DOCS_PATH = os.path.join(INDEX_DIR, "documents.pkl")

MODEL_NAME = "../bge-small-zh-v1.5"  


def load_medical_data(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = eval(line)
            if isinstance(data, tuple):
                data = data[0]

            records.append(data)
    return records


def to_text(record):
    name = record.get("name", "")
    desc = record.get("desc", "")
    cause = record.get("cause", "")
    prevent = record.get("prevent", "")
    cure_lasttime = record.get("cure_lasttime", "")
    cured_prob = record.get("cured_prob", "")
    easy_get = record.get("easy_get", "")

    common_drug = "、".join(record.get("common_drug", []))
    recommand_drug = "、".join(record.get("recommand_drug", []))
    do_eat = "、".join(record.get("do_eat", []))
    not_eat = "、".join(record.get("not_eat", []))
    check = "、".join(record.get("check", []))
    cure_department = "、".join(record.get("cure_department", []))
    symptom = "、".join(record.get("symptom", []))
    acompany = "、".join(record.get("acompany", []))
    drug_detail = "、".join(record.get("drug_detail", []))

    cure_way = "、".join(
        item[0] if isinstance(item, list) and item else item
        for item in record.get("cure_way", [])
        if item
    )

    return f"""疾病名称：{name}
疾病简介：{desc}
疾病病因：{cause}
预防措施：{prevent}
治疗周期：{cure_lasttime}
治愈概率：{cured_prob}
易感人群：{easy_get}
常用药品：{common_drug}
推荐药品：{recommand_drug}
宜吃食物：{do_eat}
忌吃食物：{not_eat}
检查项目：{check}
所属科室：{cure_department}
疾病症状：{symptom}
治疗方法：{cure_way}
并发疾病：{acompany}
药品详情：{drug_detail}
""".strip()


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("Loading medical data...")
    records = load_medical_data(DATA_PATH)

    documents = []
    for record in records:
        documents.append({
            "name": record.get("name", ""),
            "text": to_text(record),
            "record": record,
        })

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding documents...")
    embeddings = model.encode(
        [doc["text"] for doc in documents],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print("Saving index...")
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

    print(f"Done. Saved to {INDEX_DIR}")


if __name__ == "__main__":
    main()