import os
import re
import faiss
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


INTENT_TO_FIELD = {
    "查询疾病简介": "desc",
    "查询疾病病因": "cause",
    "查询疾病预防措施": "prevent",
    "查询疾病治疗周期": "cure_lasttime",
    "查询治愈概率": "cured_prob",
    "查询疾病易感人群": "easy_get",
    "查询疾病所需药品": "drug",
    "查询疾病宜吃食物": "do_eat",
    "查询疾病忌吃食物": "not_eat",
    "查询疾病所需检查项目": "check",
    "查询疾病所属科目": "cure_department",
    "查询疾病的症状": "symptom",
    "查询疾病的治疗方法": "cure_way",
    "查询疾病的并发疾病": "acompany",
    "查询药品的生产商": "drug_detail",
}


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = masked_embeddings.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class MedicalRetriever:
    def __init__(
        self,
        index_path="vector_store/medical.index",
        docs_path="vector_store/documents.pkl",
        model_path="../bge-small-zh-v1.5",
        top_k=8,
    ):
        self.index_path = index_path
        self.docs_path = docs_path
        self.model_path = model_path
        self.top_k = top_k

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.index = faiss.read_index(self.index_path)

        with open(self.docs_path, "rb") as f:
            self.documents = pickle.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

    def encode(self, texts, batch_size=16):
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                embeddings = mean_pooling(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def search_documents(self, query, top_k=None):
        top_k = top_k or self.top_k
        query_vec = self.encode(query)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            results.append(
                {
                    "score": float(score),
                    "doc_id": int(idx),
                    "doc": self.documents[idx],
                }
            )
        return results

    def _get_disease_entity(self, entities):
        return entities.get("疾病", "")

    def _get_drug_entity(self, entities):
        return entities.get("药品", "")

    def _field_has_value(self, record, intent):
        field = INTENT_TO_FIELD.get(intent)
        if not field:
            return False

        if field == "drug":
            return bool(record.get("common_drug") or record.get("recommand_drug"))
        value = record.get(field)
        return bool(value)

    def _parse_drug_producers(self, record, drug_name):
        results = []
        for item in record.get("drug_detail", []):
            if not isinstance(item, str) or "," not in item:
                continue
            producer, drug = item.split(",", 1)
            producer = producer.strip()
            drug = drug.strip()
            if drug_name and drug_name in drug:
                results.append(producer)
        return list(dict.fromkeys(results))

    def rerank_documents(self, candidates, query, entities, intent):
        disease = self._get_disease_entity(entities)
        drug = self._get_drug_entity(entities)

        reranked = []
        for item in candidates:
            score = item["score"]
            doc = item["doc"]
            record = doc["record"]
            name = doc.get("name", "") or record.get("name", "")

            bonus = 0.0

            # 1) 疾病实体精确匹配：强加权
            if disease:
                if name == disease:
                    bonus += 2.5
                elif disease in name or name in disease:
                    bonus += 1.2

            # 2) 药品生产商问题：包含目标药品时加权
            if intent == "查询药品的生产商" and drug:
                producers = self._parse_drug_producers(record, drug)
                if producers:
                    bonus += 2.0

            # 3) 请求字段非空加权
            if self._field_has_value(record, intent):
                bonus += 1.0

            # 4) query 命中疾病名/症状词的轻量加权
            text = doc.get("text", "")
            if disease and disease in text:
                bonus += 0.5
            if drug and drug in text:
                bonus += 0.5
            if query and query[:10] in text:
                bonus += 0.2

            reranked.append(
                {
                    **item,
                    "rerank_score": score + bonus,
                }
            )

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked

    def build_query_for_intent(self, raw_query, entities, intent):
        disease = self._get_disease_entity(entities)
        drug = self._get_drug_entity(entities)

        if intent == "查询药品的生产商" and drug:
            return f"{drug} 生产商 药品详情"

        if disease:
            return f"{disease} {intent}"

        if drug:
            return f"{drug} {intent}"

        return f"{raw_query} {intent}"

    def extract_field_content(self, record, intent, entities):
        if intent == "查询疾病简介":
            return record.get("desc", "")
        if intent == "查询疾病病因":
            return record.get("cause", "")
        if intent == "查询疾病预防措施":
            return record.get("prevent", "")
        if intent == "查询疾病治疗周期":
            return record.get("cure_lasttime", "")
        if intent == "查询治愈概率":
            return record.get("cured_prob", "")
        if intent == "查询疾病易感人群":
            return record.get("easy_get", "")
        if intent == "查询疾病所需药品":
            values = []
            values.extend(record.get("common_drug", []))
            values.extend(record.get("recommand_drug", []))
            values = [v for v in values if v]
            return "、".join(list(dict.fromkeys(values)))
        if intent == "查询疾病宜吃食物":
            return "、".join(record.get("do_eat", []))
        if intent == "查询疾病忌吃食物":
            return "、".join(record.get("not_eat", []))
        if intent == "查询疾病所需检查项目":
            return "、".join(record.get("check", []))
        if intent == "查询疾病所属科目":
            return "、".join(record.get("cure_department", []))
        if intent == "查询疾病的症状":
            return "、".join(record.get("symptom", []))
        if intent == "查询疾病的治疗方法":
            cure_way = []
            for item in record.get("cure_way", []):
                if isinstance(item, list):
                    if item:
                        cure_way.append(str(item[0]))
                elif item:
                    cure_way.append(str(item))
            return "、".join(cure_way)
        if intent == "查询疾病的并发疾病":
            return "、".join(record.get("acompany", []))
        if intent == "查询药品的生产商":
            drug = self._get_drug_entity(entities)
            producers = self._parse_drug_producers(record, drug)
            return "、".join(producers)

        return ""

    def retrieve_by_intent(self, raw_query, entities, intent, top_k=None):
        query = self.build_query_for_intent(raw_query, entities, intent)
        candidates = self.search_documents(query, top_k=top_k or self.top_k)
        reranked = self.rerank_documents(candidates, raw_query, entities, intent)

        for item in reranked:
            record = item["doc"]["record"]
            content = self.extract_field_content(record, intent, entities)
            if content:
                return {
                    "intent": intent,
                    "query": query,
                    "content": content,
                    "record": record,
                    "doc": item["doc"],
                    "score": item["rerank_score"],
                }

        return None

    def infer_disease_from_symptom(self, symptom, raw_query="", top_k=10):
        query = f"{symptom} 疾病症状 疾病"
        candidates = self.search_documents(query, top_k=top_k)

        scored = []
        for item in candidates:
            record = item["doc"]["record"]
            name = record.get("name", "")
            symptoms = record.get("symptom", [])
            score = item["score"]

            bonus = 0.0
            if symptom in "".join(symptoms):
                bonus += 2.0
            if raw_query:
                overlap = sum(1 for ch in set(raw_query) if ch in "".join(symptoms))
                bonus += min(overlap * 0.05, 0.5)

            scored.append((score + bonus, name))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _, name in scored:
            if name and name not in results:
                results.append(name)

        return results[:5]

    def retrieve_knowledge(self, raw_query, entities, intents):
        """
        intents: list[str]
        return:
            {
                "knowledge_blocks": [...],
                "best_record": {...} or None
            }
        """
        knowledge_blocks = []
        best_record = None

        for intent in intents:
            result = self.retrieve_by_intent(raw_query, entities, intent)
            if not result:
                continue

            if best_record is None:
                best_record = result["record"]

            entity_name = (
                entities.get("疾病")
                or entities.get("药品")
                or result["record"].get("name", "目标实体")
            )

            block = (
                f"<提示>用户对{entity_name}可能有{intent}需求，知识库内容如下："
                f"{result['content']}</提示>"
            )
            knowledge_blocks.append(
                {
                    "intent": intent,
                    "content": result["content"],
                    "prompt_block": block,
                    "score": result["score"],
                    "record_name": result["record"].get("name", ""),
                }
            )

        return {
            "knowledge_blocks": knowledge_blocks,
            "best_record": best_record,
        }


def parse_intents_from_response(response_text):
    """
    从意图识别大模型输出中提取项目内定义的意图列表
    """
    all_intents = list(INTENT_TO_FIELD.keys())
    found = []
    for intent in all_intents:
        if intent in response_text and intent not in found:
            found.append(intent)
    return found