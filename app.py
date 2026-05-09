from __future__ import annotations

import re
from itertools import combinations
from io import BytesIO, StringIO
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from wordcloud import WordCloud

try:
    import jieba
except ImportError:  # pragma: no cover - runtime fallback for minimal installs
    jieba = None


APP_TITLE = "LDA 主题建模"
APP_VERSION = "2026-05-10.1"
DEFAULT_STOPWORDS = """
的
了
和
是
在
就
都
而
及
与
着
或
一个
没有
我们
你们
他们
以及
the
and
is
are
to
of
in
for
on
with
as
by
be
this
that
from
it
an
or
at
we
they
their
our
has
have
was
were
""".strip()

DEFAULT_REPLACEMENTS = """
人工智能=>AI
新能源汽车=>新能源车
短视频=>短视频平台
electric=>ev
vehicles=>vehicle
""".strip()

LANGUAGE_MODES = {
    "自动识别": "auto",
    "中文": "zh",
    "英文": "en",
    "中英混合": "mixed",
}

WORDCLOUD_STYLES = {
    "参考图蓝橙": {
        "mode": "reference",
        "colors": ("#064b78", "#1f77b4", "#5aa7d6", "#c87f00", "#d89a27"),
    },
    "经典蓝色": {"mode": "colormap", "colormap": "Blues"},
    "活力彩色": {"mode": "colormap", "colormap": "turbo"},
    "蓝绿": {"mode": "colormap", "colormap": "winter"},
    "暖色": {"mode": "colormap", "colormap": "autumn"},
    "紫红": {"mode": "colormap", "colormap": "plasma"},
    "自然绿": {"mode": "colormap", "colormap": "Greens"},
    "灰度": {"mode": "colormap", "colormap": "Greys"},
}

POSITIVE_WORDS = {
    "好",
    "优秀",
    "提升",
    "增长",
    "改善",
    "支持",
    "喜欢",
    "满意",
    "成功",
    "创新",
    "有效",
    "便利",
    "稳定",
    "积极",
    "热门",
    "good",
    "great",
    "excellent",
    "positive",
    "better",
    "best",
    "improve",
    "improved",
    "growth",
    "success",
    "successful",
    "support",
    "satisfied",
    "love",
    "like",
    "effective",
    "efficient",
    "innovation",
    "innovative",
}

NEGATIVE_WORDS = {
    "差",
    "坏",
    "下降",
    "减少",
    "问题",
    "风险",
    "失败",
    "困难",
    "担忧",
    "不满",
    "薄弱",
    "重复",
    "慢",
    "低",
    "消极",
    "bad",
    "poor",
    "negative",
    "worse",
    "worst",
    "decline",
    "decrease",
    "risk",
    "failed",
    "failure",
    "problem",
    "issue",
    "difficult",
    "concern",
    "weak",
    "slow",
    "low",
}

NEGATION_WORDS = {"不", "没", "没有", "无", "非", "未", "not", "no", "never", "none", "without"}
INTENSIFIER_WORDS = {"很", "非常", "更", "最", "较", "十分", "特别", "very", "more", "most", "quite", "highly"}

SAMPLE_TEXT = """
人工智能正在改变教育行业，智能批改、个性化学习和课堂分析让教师能更快发现学生的薄弱环节。
许多学校开始使用学习平台收集作业数据，并根据学生表现推荐练习题。
在线教育平台通过推荐算法为学生规划课程路径，也为家长提供学习报告。
新能源汽车市场增长迅速，电池技术、充电网络和续航能力成为消费者关注的重点。
车企正在加大自动驾驶研发投入，智能座舱和辅助驾驶功能逐渐成为新车卖点。
充电桩建设速度影响新能源汽车用户体验，城市和高速服务区都在扩展基础设施。
社区医院加强慢病管理，通过随访、健康档案和远程问诊提升基层医疗服务能力。
可穿戴设备可以记录心率、睡眠和运动数据，帮助医生了解患者日常健康状况。
医疗机构正在探索人工智能辅助影像诊断，提高筛查效率并减少医生重复劳动。
文旅城市通过夜间经济、博物馆活动和特色街区吸引年轻游客。
短视频平台让小众目的地获得曝光，民宿、餐饮和交通服务也随之增长。
游客越来越重视沉浸式体验，地方文化、非遗表演和城市漫步成为热门内容。
Artificial intelligence is changing education through automated grading, personalized learning paths, and classroom analytics.
Electric vehicles depend on battery innovation, charging networks, and better driving range to improve user adoption.
Community hospitals use remote consultation, health records, and follow-up systems to manage chronic diseases.
Tourism platforms and short videos help smaller cities attract visitors through cultural events and immersive travel experiences.
""".strip()


def configure_page() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.8rem;
            max-width: 1200px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #e7e2d8;
            border-radius: 8px;
            padding: 14px 16px;
            background: #fffdf8;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def split_documents(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def read_uploaded_file(uploaded_file) -> tuple[pd.DataFrame | None, list[str]]:
    if uploaded_file is None:
        return None, []

    suffix = uploaded_file.name.lower().rsplit(".", 1)[-1]
    if suffix == "txt":
        raw = uploaded_file.read().decode("utf-8", errors="ignore")
        return None, split_documents(raw)

    if suffix == "csv":
        data = uploaded_file.read().decode("utf-8-sig", errors="ignore")
        return pd.read_csv(StringIO(data)), []

    if suffix in {"xlsx", "xls"}:
        return pd.read_excel(BytesIO(uploaded_file.read())), []

    st.warning("暂时只支持 .txt、.csv、.xlsx 和 .xls 文件。")
    return None, []


def parse_stopwords(raw_stopwords: str) -> set[str]:
    words: set[str] = set()
    for item in re.split(r"[\n,，;；\s]+", raw_stopwords):
        cleaned = item.strip().lower()
        if cleaned:
            words.add(cleaned)
    return words


def expand_stopwords(stopwords: set[str]) -> set[str]:
    expanded = set(stopwords)
    for word in stopwords:
        expanded.add(simple_english_stem(word))
    return expanded


def apply_phrase_replacements(text: str, replacements: dict[str, str]) -> str:
    normalized_text = text.lower()
    for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        if is_english_token(source):
            normalized_text = re.sub(rf"\b{re.escape(source)}\b", target, normalized_text)
        elif is_chinese_token(source):
            normalized_text = normalized_text.replace(source, target)
    return normalized_text


def apply_token_replacements(tokens: list[str], replacements: dict[str, str]) -> list[str]:
    replaced_tokens = []
    for token in tokens:
        replaced_tokens.append(replacements.get(token, token))
    return replaced_tokens


def read_wordlist_upload(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    suffix = uploaded_file.name.lower().rsplit(".", 1)[-1]
    if suffix == "txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if suffix == "csv":
        data = uploaded_file.read().decode("utf-8-sig", errors="ignore")
        dataframe = pd.read_csv(StringIO(data), header=None)
    elif suffix in {"xlsx", "xls"}:
        dataframe = pd.read_excel(BytesIO(uploaded_file.read()), header=None)
    else:
        st.warning("词表文件暂时只支持 .txt、.csv、.xlsx 和 .xls。")
        return ""

    rows = []
    for _, row in dataframe.dropna(how="all").iterrows():
        values = [str(value).strip() for value in row.dropna().tolist() if str(value).strip()]
        if not values:
            continue
        if values[0].lower() in {"原词", "旧词", "source", "from", "old"}:
            continue
        if len(values) >= 2:
            rows.append(f"{values[0]}=>{values[1]}")
        else:
            rows.append(values[0])
    return "\n".join(rows)


def combine_textarea_and_upload(textarea_value: str, uploaded_file) -> str:
    uploaded_text = read_wordlist_upload(uploaded_file)
    if textarea_value.strip() and uploaded_text.strip():
        return f"{textarea_value.strip()}\n{uploaded_text.strip()}"
    return textarea_value.strip() or uploaded_text.strip()


def parse_replacements(raw_replacements: str) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for line in raw_replacements.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        if "=>" in cleaned:
            source, target = cleaned.split("=>", 1)
        elif "=" in cleaned:
            source, target = cleaned.split("=", 1)
        elif "," in cleaned:
            source, target = cleaned.split(",", 1)
        elif "\t" in cleaned:
            source, target = cleaned.split("\t", 1)
        else:
            continue

        source = source.strip().lower()
        target = target.strip().lower()
        if source and target:
            replacements[source] = target
    return replacements


def detect_language_mode(text: str) -> str:
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_chars = len(re.findall(r"[a-zA-Z]", text))
    if chinese_chars and english_chars:
        return "mixed"
    if chinese_chars:
        return "zh"
    if english_chars:
        return "en"
    return "mixed"


def is_chinese_token(token: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", token))


def is_english_token(token: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9_'-]*", token))


def tokenize(text: str, stopwords: set[str], language_mode: str) -> list[str]:
    lowered = text.lower()
    active_mode = detect_language_mode(lowered) if language_mode == "auto" else language_mode
    if active_mode in {"zh", "mixed"} and jieba is not None:
        candidates: Iterable[str] = jieba.lcut(lowered)
    else:
        candidates = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z][a-zA-Z0-9_]+", lowered)

    tokens: list[str] = []
    for token in candidates:
        cleaned = token.strip()
        if not cleaned or cleaned in stopwords:
            continue
        if re.fullmatch(r"\d+", cleaned):
            continue
        if len(cleaned) < 2 and not re.fullmatch(r"[\u4e00-\u9fff]", cleaned):
            continue
        if active_mode == "zh" and not is_chinese_token(cleaned):
            continue
        if active_mode == "en" and not is_english_token(cleaned):
            continue
        tokens.append(cleaned)
    return tokens


def simple_english_stem(token: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]+", token):
        return token
    irregulars = {
        "movies": "movie",
    }
    if token in irregulars:
        return irregulars[token]
    for suffix in ("ization", "ational", "fulness", "ousness", "iveness", "tional", "ingly", "edly", "ing", "ed", "ies", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            if suffix == "ies":
                return f"{token[:-3]}y"
            if suffix == "es":
                return token[:-2] if token.endswith(("ses", "xes", "zes", "ches", "shes")) else token[:-1]
            return token[: -len(suffix)]
    return token


def preprocess_documents(
    documents: Iterable[str],
    stopwords: set[str],
    replacements: dict[str, str],
    language_mode: str,
    min_token_length: int,
    deduplicate_tokens: bool,
    replace_stem: bool,
) -> tuple[str, ...]:
    stopwords = expand_stopwords(stopwords)
    if jieba is not None:
        for target in replacements.values():
            if is_chinese_token(target):
                jieba.add_word(target, freq=10_000_000)

    processed_docs = []
    for document in documents:
        normalized_document = apply_phrase_replacements(document, replacements)
        tokens = tokenize(normalized_document, stopwords, language_mode)
        tokens = apply_token_replacements(tokens, replacements)
        if replace_stem:
            tokens = [simple_english_stem(token) for token in tokens]
            tokens = apply_token_replacements(tokens, replacements)
        tokens = [token for token in tokens if token not in stopwords]
        tokens = [token for token in tokens if len(token) > min_token_length]
        if deduplicate_tokens:
            tokens = list(dict.fromkeys(tokens))
        processed_docs.append(" ".join(tokens))
    return tuple(processed_docs)


def whitespace_analyzer(text: str) -> list[str]:
    return [token for token in text.split() if token]


def get_token_frequencies(prepared_documents: tuple[str, ...]) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for document in prepared_documents:
        for token in document.split():
            counts[token] = counts.get(token, 0) + 1

    frequency_df = (
        pd.DataFrame(counts.items(), columns=["词语", "词频"])
        .sort_values(["词频", "词语"], ascending=[False, True])
        .reset_index(drop=True)
    )
    frequency_df.insert(0, "排名", range(1, len(frequency_df) + 1))
    return frequency_df


def analyze_sentiment(documents: list[str], prepared_documents: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for doc_index, (original, prepared) in enumerate(zip(documents, prepared_documents), start=1):
        tokens = prepared.split()
        raw_score = 0.0
        positive_hits = 0
        negative_hits = 0
        for index, token in enumerate(tokens):
            weight = 0.0
            if token in POSITIVE_WORDS:
                weight = 1.0
                positive_hits += 1
            elif token in NEGATIVE_WORDS:
                weight = -1.0
                negative_hits += 1
            if weight == 0:
                continue

            window = tokens[max(0, index - 2) : index]
            if any(word in NEGATION_WORDS for word in window):
                weight *= -1
            if any(word in INTENSIFIER_WORDS for word in window):
                weight *= 1.5
            raw_score += weight

        normalized_score = raw_score / max(np.sqrt(len(tokens)), 1.0)
        if normalized_score > 0.08:
            label = "积极"
        elif normalized_score < -0.08:
            label = "消极"
        else:
            label = "中性"
        rows.append(
            {
                "文档序号": doc_index,
                "情感倾向": label,
                "情感得分": normalized_score,
                "积极词数": positive_hits,
                "消极词数": negative_hits,
                "有效词数": len(tokens),
                "文档": original,
            }
        )
    return pd.DataFrame(rows)


def build_cooccurrence_network(
    prepared_documents: tuple[str, ...],
    top_node_count: int,
    min_edge_weight: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frequency_df = get_token_frequencies(prepared_documents)
    top_terms = set(frequency_df.head(top_node_count)["词语"].tolist())
    edge_counts: dict[tuple[str, str], int] = {}

    for document in prepared_documents:
        terms = sorted(set(token for token in document.split() if token in top_terms))
        for left, right in combinations(terms, 2):
            edge_counts[(left, right)] = edge_counts.get((left, right), 0) + 1

    graph = nx.Graph()
    for _, row in frequency_df.head(top_node_count).iterrows():
        graph.add_node(row["词语"], frequency=int(row["词频"]))
    for (left, right), weight in edge_counts.items():
        if weight >= min_edge_weight:
            graph.add_edge(left, right, weight=weight)

    if graph.number_of_nodes() == 0:
        return pd.DataFrame(), pd.DataFrame()

    isolated_nodes = [node for node, degree in graph.degree() if degree == 0]
    graph.remove_nodes_from(isolated_nodes)
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(), pd.DataFrame()

    positions = nx.spring_layout(graph, seed=42, weight="weight", k=0.8)
    degree_centrality = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    communities = nx.community.greedy_modularity_communities(graph, weight="weight")
    community_map = {}
    for community_index, community in enumerate(communities, start=1):
        for node in community:
            community_map[node] = f"社群 {community_index}"

    node_rows = []
    for node, attrs in graph.nodes(data=True):
        node_rows.append(
            {
                "词语": node,
                "词频": attrs["frequency"],
                "度数": int(graph.degree(node)),
                "度中心性": degree_centrality.get(node, 0.0),
                "中介中心性": betweenness.get(node, 0.0),
                "社群": community_map.get(node, "社群 1"),
                "x": positions[node][0],
                "y": positions[node][1],
            }
        )

    edge_rows = []
    for left, right, attrs in graph.edges(data=True):
        edge_rows.append({"源词": left, "目标词": right, "共现次数": int(attrs["weight"])})

    node_df = pd.DataFrame(node_rows).sort_values(["度数", "词频"], ascending=[False, False])
    edge_df = pd.DataFrame(edge_rows).sort_values("共现次数", ascending=False)
    return node_df, edge_df


def find_wordcloud_font() -> str | None:
    font_paths = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in font_paths:
        if Path(font_path).exists():
            return font_path
    return None


def make_circle_mask(size: int) -> np.ndarray:
    y, x = np.ogrid[:size, :size]
    center = (size - 1) / 2
    radius = size * 0.485
    mask = ((x - center) ** 2 + (y - center) ** 2 > radius**2) * 255
    return mask.astype(np.uint8)


@st.cache_data(show_spinner=False)
def build_wordcloud_image(
    frequencies: tuple[tuple[str, int], ...],
    style_name: str,
    max_words: int,
) -> bytes:
    font_path = find_wordcloud_font()
    style = WORDCLOUD_STYLES.get(style_name, WORDCLOUD_STYLES["参考图蓝橙"])
    color_func = None
    colormap = style.get("colormap", "Blues")
    background_color = "white"

    if style["mode"] == "reference":
        colors = style["colors"]

        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            if font_size >= 105:
                palette = colors[:2] + (colors[3],)
            elif font_size >= 55:
                palette = colors[:4]
            else:
                palette = colors
            return palette[sum(ord(char) for char in word) % len(palette)]

    canvas_size = 1160
    wordcloud = WordCloud(
        width=canvas_size,
        height=canvas_size,
        background_color=background_color,
        colormap=colormap,
        color_func=color_func,
        max_words=max_words,
        prefer_horizontal=0.97,
        relative_scaling=0.72,
        max_font_size=260,
        min_font_size=7,
        random_state=42,
        font_path=font_path,
        collocations=False,
        margin=0,
        repeat=True,
        mask=make_circle_mask(canvas_size),
        scale=2,
    ).generate_from_frequencies(dict(frequencies))

    output = BytesIO()
    wordcloud.to_image().save(output, format="PNG")
    return output.getvalue()


@st.cache_data(show_spinner=False)
def fit_lda(
    prepared_documents: tuple[str, ...],
    n_topics: int,
    max_features: int,
    min_df: int,
    max_df: float,
    max_iter: int,
    random_state: int,
    alpha: str,
    beta: str,
    top_n_words: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int, float, float]:
    doc_topic_prior = None if alpha == "auto" else float(alpha)
    topic_word_prior = None if beta == "auto" else float(beta)
    vectorizer = CountVectorizer(
        analyzer=whitespace_analyzer,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )
    term_matrix = vectorizer.fit_transform(prepared_documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        max_iter=max_iter,
        doc_topic_prior=doc_topic_prior,
        topic_word_prior=topic_word_prior,
        random_state=random_state,
        evaluate_every=-1,
    )
    doc_topic = lda.fit_transform(term_matrix)
    feature_names = np.array(vectorizer.get_feature_names_out())
    term_frequency = np.asarray(term_matrix.sum(axis=0)).ravel()
    term_frequency_df = (
        pd.DataFrame({"词语": feature_names, "全局词频": term_frequency})
        .sort_values(["全局词频", "词语"], ascending=[False, True])
        .reset_index(drop=True)
    )

    topic_rows = []
    for topic_index, weights in enumerate(lda.components_, start=1):
        top_indices = weights.argsort()[::-1][:top_n_words]
        total = weights[top_indices].sum()
        for rank, term_index in enumerate(top_indices, start=1):
            weight = float(weights[term_index])
            topic_rows.append(
                {
                    "主题": f"主题 {topic_index}",
                    "排名": rank,
                    "关键词": feature_names[term_index],
                    "权重": weight,
                    "主题内占比": weight / total if total else 0,
                }
            )

    topic_df = pd.DataFrame(topic_rows)
    dist_df = pd.DataFrame(
        doc_topic,
        columns=[f"主题 {i}" for i in range(1, n_topics + 1)],
    )
    dist_df.insert(0, "主导主题", dist_df.iloc[:, :].idxmax(axis=1))
    dist_df.insert(0, "文档序号", range(1, len(prepared_documents) + 1))

    topic_columns = [column for column in dist_df.columns if column.startswith("主题 ")]
    topic_share = (
        dist_df[topic_columns]
        .mean()
        .reset_index()
        .rename(columns={"index": "主题", 0: "平均占比"})
    )
    topic_word = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
    topic_word_df = pd.DataFrame(topic_word, columns=feature_names)
    bubble_df = make_bubble_df(topic_word, topic_share)
    perplexity = float(lda.perplexity(term_matrix))
    log_likelihood = float(lda.score(term_matrix))
    return topic_df, dist_df, topic_share, topic_word_df, bubble_df, term_frequency_df, int(term_matrix.shape[1]), perplexity, log_likelihood


def make_bubble_df(topic_word: np.ndarray, topic_share: pd.DataFrame) -> pd.DataFrame:
    if topic_word.shape[0] == 1:
        coords = np.array([[0.0, 0.0]])
    else:
        distances = cosine_distances(topic_word)
        centered = distances - distances.mean(axis=0)
        u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        if u.shape[1] == 1:
            coords = np.column_stack([u[:, 0] * singular_values[0], np.zeros(topic_word.shape[0])])
        else:
            coords = u[:, :2] * singular_values[:2]

    bubble_df = topic_share.copy()
    bubble_df["x"] = coords[:, 0]
    bubble_df["y"] = coords[:, 1]
    bubble_df["气泡大小"] = bubble_df["平均占比"] * 100
    return bubble_df


def approximate_coherence(prepared_documents: tuple[str, ...], topic_df: pd.DataFrame, top_n: int) -> float:
    doc_sets = [set(document.split()) for document in prepared_documents if document.strip()]
    scores = []
    for _, group in topic_df.groupby("主题"):
        words = group.sort_values("排名")["关键词"].head(top_n).tolist()
        pair_scores = []
        for i, left in enumerate(words):
            left_count = sum(left in doc for doc in doc_sets)
            for right in words[i + 1 :]:
                both_count = sum(left in doc and right in doc for doc in doc_sets)
                pair_scores.append(np.log((both_count + 1) / (left_count + 1)))
        if pair_scores:
            scores.append(float(np.mean(pair_scores)))
    return float(np.mean(scores)) if scores else 0.0


@st.cache_data(show_spinner=False)
def evaluate_topic_counts(
    prepared_documents: tuple[str, ...],
    topic_counts: tuple[int, ...],
    max_features: int,
    min_df: int,
    max_df: float,
    max_iter: int,
    random_state: int,
    alpha: str,
    beta: str,
    top_n_words: int,
) -> pd.DataFrame:
    rows = []
    for topic_count in topic_counts:
        topic_df, _, _, _, _, _, vocab_size, perplexity, log_likelihood = fit_lda(
            prepared_documents,
            topic_count,
            max_features,
            min_df,
            max_df,
            max_iter,
            random_state,
            alpha,
            beta,
            top_n_words,
        )
        rows.append(
            {
                "主题数": topic_count,
                "c_v一致性": approximate_coherence(prepared_documents, topic_df, min(8, top_n_words)),
                "困惑度": perplexity,
                "对数似然": log_likelihood,
                "有效词表数": vocab_size,
            }
        )
    return pd.DataFrame(rows)


def load_documents() -> list[str]:
    st.subheader("输入文本")
    source = st.radio(
        "数据来源",
        ["使用示例", "粘贴文本", "上传文件"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source == "使用示例":
        st.text_area("示例文本", SAMPLE_TEXT, height=220, disabled=True)
        return split_documents(SAMPLE_TEXT)

    if source == "粘贴文本":
        text = st.text_area("每行一篇文档", height=260, placeholder="把文本粘贴到这里，每行作为一篇文档")
        return split_documents(text)

    uploaded = st.file_uploader("上传 .txt、.csv 或 Excel 文件", type=["txt", "csv", "xlsx", "xls"])
    dataframe, txt_docs = read_uploaded_file(uploaded)
    if txt_docs:
        return txt_docs
    if dataframe is not None:
        st.dataframe(dataframe.head(20), use_container_width=True)
        text_columns = [
            column
            for column in dataframe.columns
            if pd.api.types.is_string_dtype(dataframe[column]) or dataframe[column].dtype == object
        ]
        if not text_columns:
            st.error("CSV 中没有可用的文本列。")
            return []
        selected_column = st.selectbox("选择文本列", text_columns)
        return [
            str(value).strip()
            for value in dataframe[selected_column].dropna().tolist()
            if str(value).strip()
        ]
    return []


def render_topics(topic_df: pd.DataFrame) -> None:
    st.subheader("主题关键词")
    topic_names = topic_df["主题"].drop_duplicates().tolist()
    tabs = st.tabs(topic_names)
    for tab, topic_name in zip(tabs, topic_names):
        with tab:
            data = topic_df[topic_df["主题"] == topic_name].copy()
            chart = px.bar(
                data.sort_values("权重"),
                x="权重",
                y="关键词",
                orientation="h",
                color="主题内占比",
                color_continuous_scale=["#7c6f64", "#d65a31", "#4f8f7a"],
                height=430,
            )
            chart.update_layout(
                margin=dict(l=12, r=12, t=20, b=12),
                xaxis_title="权重",
                yaxis_title="",
                coloraxis_colorbar_title="占比",
            )
            st.plotly_chart(chart, use_container_width=True)
            st.dataframe(
                data[["排名", "关键词", "权重", "主题内占比"]],
                use_container_width=True,
                hide_index=True,
            )


def render_document_distribution(dist_df: pd.DataFrame, topic_share: pd.DataFrame) -> None:
    st.subheader("文档主题分布")
    left, right = st.columns([1, 1])
    with left:
        share_chart = px.pie(
            topic_share,
            values="平均占比",
            names="主题",
            hole=0.45,
            color_discrete_sequence=["#4f8f7a", "#d65a31", "#7c6f64", "#c9a227", "#5f6f94", "#9a6a7a"],
        )
        share_chart.update_layout(margin=dict(l=8, r=8, t=8, b=8), showlegend=True)
        st.plotly_chart(share_chart, use_container_width=True)
    with right:
        st.dataframe(dist_df, use_container_width=True, hide_index=True)


def select_numeric_or_auto(label: str, default: str = "auto"):
    options = {
        "默认 自动": "auto",
        "0.01": "0.01",
        "0.05": "0.05",
        "0.1": "0.1",
        "0.5": "0.5",
        "1.0": "1.0",
    }
    selected_label = st.selectbox(label, list(options.keys()), index=list(options.values()).index(default) if default in options.values() else 0)
    return options[selected_label]


def render_training_controls() -> dict:
    st.subheader("模型训练设置")
    with st.container(border=True):
        row = st.columns([1.0, 0.9, 1.0, 1.0, 1.05, 0.9, 1.0, 1.35])
        language_label = row[0].selectbox("处理语言", list(LANGUAGE_MODES.keys()), index=0)
        iterations = row[1].selectbox("全局训练次数", [5, 10, 20, 50, 100], index=2)
        with row[2]:
            alpha = select_numeric_or_auto("α")
        with row[3]:
            beta = select_numeric_or_auto("β")
        min_token_length = row[4].selectbox("去单词长度 <=", [0, 1, 2, 3, 4], index=0)
        deduplicate_tokens = row[5].selectbox("分词去重", ["否", "是"], index=0) == "是"
        replace_stem = row[6].selectbox("词干替换", ["否", "是"], index=0) == "是"
        row[7].button("读取文件加载模型", type="primary", key="train_model", use_container_width=True)

        advanced = st.expander("更多参数", expanded=False)
        with advanced:
            col_a, col_b, col_c, col_d = st.columns(4)
            n_topics = col_a.slider("默认主题数", min_value=2, max_value=20, value=5)
            max_features = col_b.slider("最大词表数", min_value=100, max_value=8000, value=1000, step=100)
            min_df = col_c.slider("词语最少出现文档数", min_value=1, max_value=10, value=1)
            max_df = col_d.slider("词语最多出现文档比例", min_value=0.50, max_value=1.00, value=0.95, step=0.05)
            random_state = st.number_input("随机种子", min_value=0, value=42, step=1)
            stop_col, replacement_col = st.columns(2)
            with stop_col:
                raw_stopwords = st.text_area("停用词", DEFAULT_STOPWORDS, height=180)
                stopword_upload = st.file_uploader(
                    "上传停用词文件",
                    type=["txt", "csv", "xlsx", "xls"],
                    key="stopword_upload",
                )
            with replacement_col:
                raw_replacements = st.text_area(
                    "替换词",
                    DEFAULT_REPLACEMENTS,
                    height=180,
                    help="每行一条，格式：原词=>新词。也支持 原词=新词、原词,新词。",
                )
                replacement_upload = st.file_uploader(
                    "上传替换词文件",
                    type=["txt", "csv", "xlsx", "xls"],
                    key="replacement_upload",
                )

            raw_stopwords = combine_textarea_and_upload(raw_stopwords, stopword_upload)
            raw_replacements = combine_textarea_and_upload(raw_replacements, replacement_upload)

    return {
        "n_topics": n_topics,
        "max_features": max_features,
        "min_df": min_df,
        "max_df": max_df,
        "max_iter": iterations,
        "random_state": int(random_state),
        "alpha": alpha,
        "beta": beta,
        "language_mode": LANGUAGE_MODES[language_label],
        "min_token_length": min_token_length,
        "deduplicate_tokens": deduplicate_tokens,
        "replace_stem": replace_stem,
        "raw_stopwords": raw_stopwords,
        "raw_replacements": raw_replacements,
    }


def prepare_current_documents(documents: list[str], settings: dict) -> tuple[str, ...]:
    return preprocess_documents(
        documents,
        parse_stopwords(settings["raw_stopwords"]),
        parse_replacements(settings["raw_replacements"]),
        settings["language_mode"],
        settings["min_token_length"],
        settings["deduplicate_tokens"],
        settings["replace_stem"],
    )


def make_classification_df(
    documents: list[str],
    prepared_documents: tuple[str, ...],
    dist_df: pd.DataFrame,
    topic_df: pd.DataFrame,
) -> pd.DataFrame:
    topic_columns = [column for column in dist_df.columns if column.startswith("主题 ")]
    topic_keywords = (
        topic_df.sort_values(["主题", "排名"])
        .groupby("主题")["关键词"]
        .apply(lambda words: "、".join(words.head(8).astype(str)))
        .to_dict()
    )
    classification_df = pd.DataFrame(
        {
            "文档序号": range(1, len(documents) + 1),
            "原始评论": documents,
            "清理后评论": list(prepared_documents),
        }
    )
    classification_df["分类主题"] = dist_df["主导主题"].to_list()
    classification_df["主题关键词"] = classification_df["分类主题"].map(topic_keywords).fillna("")
    classification_df["主题置信度"] = dist_df[topic_columns].max(axis=1).to_list()
    for topic_column in topic_columns:
        classification_df[f"{topic_column}概率"] = dist_df[topic_column].to_list()
    return classification_df.sort_values(["分类主题", "文档序号"]).reset_index(drop=True)


def train_current_model(documents: list[str], settings: dict, topic_count: int | None = None, top_n_words: int = 15) -> None:
    prepared_documents = prepare_current_documents(documents, settings)
    topic_df, dist_df, topic_share, topic_word_df, bubble_df, term_frequency_df, vocab_size, perplexity, log_likelihood = fit_lda(
        prepared_documents,
        topic_count or settings["n_topics"],
        settings["max_features"],
        settings["min_df"],
        settings["max_df"],
        settings["max_iter"],
        settings["random_state"],
        settings["alpha"],
        settings["beta"],
        top_n_words,
    )
    classification_df = make_classification_df(documents, prepared_documents, dist_df, topic_df)
    st.session_state["lda_result"] = {
        "prepared_documents": prepared_documents,
        "topic_df": topic_df,
        "dist_df": dist_df,
        "classification_df": classification_df,
        "topic_share": topic_share,
        "topic_word_df": topic_word_df,
        "bubble_df": bubble_df,
        "term_frequency_df": term_frequency_df,
        "vocab_size": vocab_size,
        "perplexity": perplexity,
        "log_likelihood": log_likelihood,
        "topic_count": topic_count or settings["n_topics"],
        "top_n_words": top_n_words,
    }


def ensure_trained(documents: list[str], settings: dict, topic_count: int, top_n_words: int) -> dict:
    result = st.session_state.get("lda_result")
    if result is None or result.get("topic_count") != topic_count or result.get("top_n_words") < top_n_words:
        train_current_model(documents, settings, topic_count, top_n_words)
    return st.session_state["lda_result"]


def render_tool_panels(documents: list[str], settings: dict) -> None:
    with st.container(border=True):
        panel_a, panel_b, panel_c, panel_d, panel_e = st.columns(5)

        with panel_a:
            st.markdown("**LDA主题数量评估**")
            rule = st.selectbox("选择评估规则", ["c_v一致性", "困惑度", "对数似然"], key="eval_rule")
            max_topics = max(2, min(20, len(documents)))
            topic_range_end = st.selectbox("评估最大主题数", list(range(2, max_topics + 1)), index=min(6, max_topics) - 2)
            run_eval = st.button("点击计算", key="run_eval", use_container_width=True)

        with panel_b:
            st.markdown("**生成主题权重词**")
            weight_topic_count = st.selectbox("选择主题数", list(range(2, min(20, len(documents)) + 1)), index=min(3, max(0, len(documents) - 2)), key="weight_topics")
            top_n_words = st.selectbox("选择主题词数", [5, 8, 10, 15, 20, 30], index=0, key="top_n_words")
            run_words = st.button("点击生成权重词", key="run_words", use_container_width=True)

        with panel_c:
            st.markdown("**LDA气泡图可视化**")
            bubble_topic_count = st.selectbox("选择主题数", list(range(2, min(20, len(documents)) + 1)), index=min(3, max(0, len(documents) - 2)), key="bubble_topics")
            run_bubble = st.button("点击生成主题可视化", key="run_bubble", use_container_width=True)

        with panel_d:
            st.markdown("**主题强度变化**")
            intensity_topic_count = st.selectbox("选择主题数", list(range(2, min(20, len(documents)) + 1)), index=min(3, max(0, len(documents) - 2)), key="intensity_topics")
            run_intensity = st.button("点击生成主题强度变化", key="run_intensity", use_container_width=True)

        with panel_e:
            st.markdown("**总词频与词云图**")
            max_cloud_words = st.selectbox("选择词云词数", [50, 100, 150, 200, 300], index=2, key="cloud_words")
            style_label = st.selectbox("选择配色", list(WORDCLOUD_STYLES.keys()), index=0, key="cloud_palette")
            run_wordcloud = st.button("点击生成词云图", key="run_wordcloud", use_container_width=True)

        panel_f, panel_g = st.columns(2)
        with panel_f:
            st.markdown("**文本情感分析**")
            run_sentiment = st.button("点击生成情感分析", key="run_sentiment", use_container_width=True)

        with panel_g:
            st.markdown("**社会网络关系图**")
            network_nodes = st.selectbox("选择节点数", [20, 30, 50, 80, 100], index=1, key="network_nodes")
            min_edge_weight = st.selectbox("最小共现次数", [1, 2, 3, 5, 8], index=0, key="network_min_edge")
            run_network = st.button("点击生成关系图", key="run_network", use_container_width=True)

        run_preview = st.button("预览预处理结果", key="run_preview", use_container_width=True)

    if run_eval:
        prepared_documents = prepare_current_documents(documents, settings)
        with st.spinner("正在评估不同主题数..."):
            eval_df = evaluate_topic_counts(
                prepared_documents,
                tuple(range(2, topic_range_end + 1)),
                settings["max_features"],
                settings["min_df"],
                settings["max_df"],
                settings["max_iter"],
                settings["random_state"],
                settings["alpha"],
                settings["beta"],
                10,
            )
        st.session_state["eval_df"] = eval_df

    if run_words:
        with st.spinner("正在生成主题权重词..."):
            ensure_trained(documents, settings, weight_topic_count, top_n_words)
        st.session_state["active_view"] = "words"

    if run_bubble:
        with st.spinner("正在生成主题气泡图..."):
            ensure_trained(documents, settings, bubble_topic_count, 15)
        st.session_state["active_view"] = "bubble"

    if run_intensity:
        with st.spinner("正在计算主题强度变化..."):
            ensure_trained(documents, settings, intensity_topic_count, 15)
        st.session_state["active_view"] = "intensity"

    if run_wordcloud:
        prepared_documents = prepare_current_documents(documents, settings)
        frequency_df = get_token_frequencies(prepared_documents)
        if frequency_df.empty:
            st.warning("当前预处理后没有可统计的词语，请调整语言模式、停用词或词长过滤。")
        else:
            with st.spinner("正在统计词频并生成词云图..."):
                frequencies = tuple(
                    (str(row["词语"]), int(row["词频"]))
                    for _, row in frequency_df.head(max_cloud_words).iterrows()
                )
                image_bytes = build_wordcloud_image(
                    frequencies,
                    style_label,
                    max_cloud_words,
                )
            st.session_state["frequency_df"] = frequency_df
            st.session_state["wordcloud_image"] = image_bytes
            st.session_state["active_view"] = "wordcloud"

    if run_sentiment:
        prepared_documents = prepare_current_documents(documents, settings)
        with st.spinner("正在生成文本情感分析..."):
            sentiment_df = analyze_sentiment(documents, prepared_documents)
        st.session_state["sentiment_df"] = sentiment_df
        st.session_state["active_view"] = "sentiment"

    if run_network:
        prepared_documents = prepare_current_documents(documents, settings)
        with st.spinner("正在生成社会网络关系图..."):
            node_df, edge_df = build_cooccurrence_network(prepared_documents, network_nodes, min_edge_weight)
        if node_df.empty or edge_df.empty:
            st.warning("当前条件下没有足够的共现关系，请降低最小共现次数或增加节点数。")
        else:
            st.session_state["network_node_df"] = node_df
            st.session_state["network_edge_df"] = edge_df
            st.session_state["active_view"] = "network"

    if run_preview:
        prepared_documents = prepare_current_documents(documents, settings)
        preview_df = pd.DataFrame(
            {
                "文档序号": range(1, len(prepared_documents) + 1),
                "预处理结果": prepared_documents,
            }
        )
        st.session_state["preview_df"] = preview_df
        st.session_state["active_view"] = "preview"

    if "eval_df" in st.session_state:
        render_evaluation(st.session_state["eval_df"], st.session_state.get("eval_rule", "c_v一致性"))

    result = st.session_state.get("lda_result")
    if result:
        metric_cols = st.columns(4)
        metric_cols[0].metric("有效词表数", result["vocab_size"])
        metric_cols[1].metric("困惑度", f"{result['perplexity']:.2f}")
        metric_cols[2].metric("对数似然", f"{result['log_likelihood']:.0f}")
        metric_cols[3].metric("当前主题数", result["topic_count"])

        active_view = st.session_state.get("active_view", "words")
        if active_view == "bubble":
            render_bubble_chart(result)
        elif active_view == "intensity":
            render_intensity_chart(result["dist_df"])
        elif active_view == "wordcloud" and "frequency_df" in st.session_state:
            render_wordcloud(st.session_state["frequency_df"], st.session_state.get("wordcloud_image"))
        elif active_view == "sentiment" and "sentiment_df" in st.session_state:
            render_sentiment(st.session_state["sentiment_df"])
        elif active_view == "network" and "network_node_df" in st.session_state:
            render_network(st.session_state["network_node_df"], st.session_state["network_edge_df"])
        elif active_view == "preview" and "preview_df" in st.session_state:
            render_preprocess_preview(st.session_state["preview_df"])
        else:
            render_topics(result["topic_df"])
            render_document_distribution(result["dist_df"], result["topic_share"])
            render_classification_table(result["classification_df"])
            render_downloads(result["topic_df"], result["dist_df"], result["classification_df"])
    elif st.session_state.get("active_view") == "wordcloud" and "frequency_df" in st.session_state:
        render_wordcloud(st.session_state["frequency_df"], st.session_state.get("wordcloud_image"))
    elif st.session_state.get("active_view") == "sentiment" and "sentiment_df" in st.session_state:
        render_sentiment(st.session_state["sentiment_df"])
    elif st.session_state.get("active_view") == "network" and "network_node_df" in st.session_state:
        render_network(st.session_state["network_node_df"], st.session_state["network_edge_df"])
    elif st.session_state.get("active_view") == "preview" and "preview_df" in st.session_state:
        render_preprocess_preview(st.session_state["preview_df"])


def render_evaluation(eval_df: pd.DataFrame, selected_rule: str) -> None:
    st.subheader("LDA主题数量评估")
    metric_column = selected_rule if selected_rule in eval_df.columns else "c_v一致性"
    chart = px.line(eval_df, x="主题数", y=metric_column, markers=True)
    chart.update_layout(margin=dict(l=12, r=12, t=20, b=12), xaxis_dtick=1)
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(eval_df, use_container_width=True, hide_index=True)


def make_relevance_terms(
    topic_word_df: pd.DataFrame,
    term_frequency_df: pd.DataFrame,
    selected_topic: int,
    lambda_value: float,
    topic_share: float,
    top_n: int = 30,
) -> pd.DataFrame:
    topic_index = selected_topic - 1
    topic_probs = topic_word_df.iloc[topic_index].astype(float)
    corpus_frequency = term_frequency_df.set_index("词语")["全局词频"].reindex(topic_word_df.columns).fillna(0).astype(float)
    corpus_probs = corpus_frequency / max(float(corpus_frequency.sum()), 1.0)
    lift = topic_probs / corpus_probs.replace(0, np.nan)
    relevance = lambda_value * np.log(topic_probs.replace(0, np.nan)) + (1 - lambda_value) * np.log(lift)

    terms_df = pd.DataFrame(
        {
            "词语": topic_word_df.columns,
            "相关度": relevance.replace([np.inf, -np.inf], np.nan).fillna(-1e9).to_numpy(),
            "主题词概率": topic_probs.to_numpy(),
            "全局词频": corpus_frequency.to_numpy(),
        }
    )
    topic_total = max(float(corpus_frequency.sum()) * topic_share, 1.0)
    terms_df["主题内估计词频"] = terms_df["主题词概率"] * topic_total
    terms_df = terms_df.sort_values("相关度", ascending=False).head(top_n).copy()
    terms_df["排名"] = range(1, len(terms_df) + 1)
    return terms_df


def select_previous_bubble_topic(topic_count: int) -> None:
    current = int(st.session_state.get("selected_bubble_topic", 0))
    st.session_state["selected_bubble_topic"] = topic_count if current <= 1 else current - 1


def select_next_bubble_topic(topic_count: int) -> None:
    current = int(st.session_state.get("selected_bubble_topic", 0))
    st.session_state["selected_bubble_topic"] = 1 if current >= topic_count else current + 1


def clear_bubble_topic() -> None:
    st.session_state["selected_bubble_topic"] = 0


def render_bubble_chart(result: dict) -> None:
    bubble_df = result["bubble_df"]
    topic_word_df = result["topic_word_df"]
    term_frequency_df = result["term_frequency_df"]
    topic_count = int(result["topic_count"])

    st.subheader("LDA主题词气泡图")
    if "selected_bubble_topic" not in st.session_state:
        st.session_state["selected_bubble_topic"] = 0
    if int(st.session_state["selected_bubble_topic"]) > topic_count:
        st.session_state["selected_bubble_topic"] = 0

    controls = st.columns([1.1, 0.9, 0.9, 0.9, 2.4])
    selected_topic = controls[0].number_input("Selected Topic", min_value=0, max_value=topic_count, step=1, key="selected_bubble_topic")
    controls[1].button("Previous Topic", use_container_width=True, on_click=select_previous_bubble_topic, args=(topic_count,))
    controls[2].button("Next Topic", use_container_width=True, on_click=select_next_bubble_topic, args=(topic_count,))
    controls[3].button("Clear Topic", use_container_width=True, on_click=clear_bubble_topic)
    lambda_value = controls[4].slider("Slide to adjust relevance metric", 0.0, 1.0, 1.0, 0.1)

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("**Intertopic Distance Map (via multidimensional scaling)**")
        plot_df = bubble_df.copy()
        plot_df["主题编号"] = plot_df["主题"].str.extract(r"(\d+)").astype(int)
        plot_df["选中"] = np.where(plot_df["主题编号"] == selected_topic, "Selected", "Topic")
        distance_chart = px.scatter(
            plot_df,
            x="x",
            y="y",
            size="气泡大小",
            color="选中",
            text="主题编号",
            size_max=95,
            color_discrete_map={"Topic": "#bcd7e8", "Selected": "#d95f5f"},
            hover_data={"主题": True, "平均占比": ":.2%", "x": False, "y": False, "气泡大小": False, "主题编号": False, "选中": False},
        )
        distance_chart.update_traces(textposition="middle center", marker=dict(line=dict(width=1, color="#7f8f99"), opacity=0.72))
        distance_chart.add_hline(y=0, line_width=1, line_color="#d8d8d8")
        distance_chart.add_vline(x=0, line_width=1, line_color="#d8d8d8")
        distance_chart.update_layout(
            height=590,
            margin=dict(l=12, r=12, t=8, b=8),
            xaxis_title="PC1",
            yaxis_title="PC2",
            showlegend=False,
        )
        st.plotly_chart(distance_chart, use_container_width=True)
        st.caption("气泡面积表示主题在语料中的平均占比。")

    with right:
        if selected_topic == 0:
            terms_df = term_frequency_df.head(30).copy()
            terms_df["主题内估计词频"] = 0.0
            terms_df.insert(0, "排名", range(1, len(terms_df) + 1))
            title = "Top-30 Most Salient Terms"
        else:
            selected_share = float(
                bubble_df.loc[bubble_df["主题"] == f"主题 {selected_topic}", "平均占比"].iloc[0]
            )
            terms_df = make_relevance_terms(topic_word_df, term_frequency_df, selected_topic, lambda_value, selected_share, 30)
            title = f"Top-30 Relevant Terms for Topic {selected_topic}"

        st.markdown(f"**{title}**")
        bar_df = terms_df.sort_values("全局词频", ascending=True)
        bar_chart = go.Figure()
        bar_chart.add_trace(
            go.Bar(
                x=bar_df["全局词频"],
                y=bar_df["词语"],
                orientation="h",
                name="Overall term frequency",
                marker_color="#a8cfe3",
            )
        )
        if selected_topic != 0:
            bar_chart.add_trace(
                go.Bar(
                    x=bar_df["主题内估计词频"],
                    y=bar_df["词语"],
                    orientation="h",
                    name="Estimated term frequency within selected topic",
                    marker_color="#d95f5f",
                )
            )
        bar_chart.update_layout(
            barmode="overlay",
            height=590,
            margin=dict(l=12, r=12, t=8, b=58),
            xaxis_title="词频",
            yaxis_title="",
            legend=dict(orientation="h", y=-0.12, x=0),
        )
        st.plotly_chart(bar_chart, use_container_width=True)

    st.dataframe(terms_df, use_container_width=True, hide_index=True)


def render_intensity_chart(dist_df: pd.DataFrame) -> None:
    st.subheader("主题强度变化")
    topic_columns = [column for column in dist_df.columns if column.startswith("主题 ")]
    if not topic_columns:
        st.info("暂无主题强度数据。")
        return
    chart_df = dist_df[["文档序号", *topic_columns]].melt("文档序号", var_name="主题", value_name="强度")
    chart = px.line(
        chart_df,
        x="文档序号",
        y="强度",
        color="主题",
        markers=True,
        color_discrete_sequence=["#4f8f7a", "#d65a31", "#7c6f64", "#c9a227", "#5f6f94", "#9a6a7a"],
    )
    chart.update_layout(height=480, margin=dict(l=12, r=12, t=20, b=12), yaxis_tickformat=".0%")
    st.plotly_chart(chart, use_container_width=True)
    st.caption("强度变化按当前文档顺序绘制；如果 CSV 已按年份或时间排序，可用于观察主题随时间的变化。")


def render_wordcloud(frequency_df: pd.DataFrame, image_bytes: bytes | None) -> None:
    st.subheader("总词频统计与词云图")
    left, right = st.columns([1.2, 1])
    with left:
        if image_bytes:
            st.image(image_bytes, use_container_width=True)
            st.download_button(
                "下载词云 PNG",
                image_bytes,
                file_name="wordcloud.png",
                mime="image/png",
                use_container_width=True,
            )
        else:
            st.info("请先点击“生成词云图”。")

    with right:
        top_words = frequency_df.head(30).sort_values("词频")
        chart = px.bar(
            top_words,
            x="词频",
            y="词语",
            orientation="h",
            color="词频",
            color_continuous_scale=["#4f8f7a", "#d65a31", "#f4d35e"],
            height=520,
        )
        chart.update_layout(margin=dict(l=12, r=12, t=20, b=12), yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(chart, use_container_width=True)

    st.dataframe(frequency_df, use_container_width=True, hide_index=True)
    st.download_button(
        "下载总词频 CSV",
        frequency_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="word_frequencies.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_sentiment(sentiment_df: pd.DataFrame) -> None:
    st.subheader("文本情感分析")
    counts = sentiment_df["情感倾向"].value_counts().reindex(["积极", "中性", "消极"], fill_value=0).reset_index()
    counts.columns = ["情感倾向", "文档数"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("平均情感得分", f"{sentiment_df['情感得分'].mean():.3f}")
    metric_cols[1].metric("积极文档", int(counts.loc[counts["情感倾向"] == "积极", "文档数"].iloc[0]))
    metric_cols[2].metric("中性文档", int(counts.loc[counts["情感倾向"] == "中性", "文档数"].iloc[0]))
    metric_cols[3].metric("消极文档", int(counts.loc[counts["情感倾向"] == "消极", "文档数"].iloc[0]))

    left, right = st.columns([0.9, 1.1])
    with left:
        pie = px.pie(
            counts,
            values="文档数",
            names="情感倾向",
            hole=0.42,
            color="情感倾向",
            color_discrete_map={"积极": "#4f8f7a", "中性": "#9ca3af", "消极": "#d95f5f"},
        )
        pie.update_layout(margin=dict(l=12, r=12, t=16, b=12))
        st.plotly_chart(pie, use_container_width=True)

    with right:
        line = px.line(
            sentiment_df,
            x="文档序号",
            y="情感得分",
            color="情感倾向",
            markers=True,
            color_discrete_map={"积极": "#4f8f7a", "中性": "#9ca3af", "消极": "#d95f5f"},
        )
        line.add_hline(y=0, line_width=1, line_dash="dash", line_color="#9ca3af")
        line.update_layout(height=390, margin=dict(l=12, r=12, t=16, b=12))
        st.plotly_chart(line, use_container_width=True)

    st.dataframe(sentiment_df, use_container_width=True, hide_index=True)
    st.download_button(
        "下载情感分析 CSV",
        sentiment_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="sentiment_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_network(node_df: pd.DataFrame, edge_df: pd.DataFrame) -> None:
    st.subheader("社会网络关系图")
    node_lookup = node_df.set_index("词语")

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for _, edge in edge_df.iterrows():
        if edge["源词"] not in node_lookup.index or edge["目标词"] not in node_lookup.index:
            continue
        source = node_lookup.loc[edge["源词"]]
        target = node_lookup.loc[edge["目标词"]]
        edge_x.extend([source["x"], target["x"], None])
        edge_y.extend([source["y"], target["y"], None])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.8, color="rgba(150, 160, 170, 0.45)"),
            hoverinfo="none",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=node_df["x"],
            y=node_df["y"],
            mode="markers+text",
            text=node_df["词语"],
            textposition="top center",
            marker=dict(
                size=np.clip(node_df["词频"] * 4 + node_df["度数"] * 3, 12, 58),
                color=node_df["度中心性"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="度中心性"),
                line=dict(width=1, color="#f6f6f6"),
            ),
            customdata=node_df[["词频", "度数", "中介中心性", "社群"]],
            hovertemplate="<b>%{text}</b><br>词频: %{customdata[0]}<br>度数: %{customdata[1]}<br>中介中心性: %{customdata[2]:.3f}<br>%{customdata[3]}<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_layout(
        height=650,
        margin=dict(l=12, r=12, t=16, b=12),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    st.plotly_chart(figure, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**节点中心性**")
        st.dataframe(
            node_df[["词语", "词频", "度数", "度中心性", "中介中心性", "社群"]],
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.markdown("**共现关系**")
        st.dataframe(edge_df, use_container_width=True, hide_index=True)

    download_cols = st.columns(2)
    download_cols[0].download_button(
        "下载节点 CSV",
        node_df.drop(columns=["x", "y"]).to_csv(index=False).encode("utf-8-sig"),
        file_name="network_nodes.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_cols[1].download_button(
        "下载关系 CSV",
        edge_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="network_edges.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_preprocess_preview(preview_df: pd.DataFrame) -> None:
    st.subheader("预处理结果预览")
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
    st.download_button(
        "下载预处理结果 CSV",
        preview_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="preprocessed_documents.csv",
        mime="text/csv",
        use_container_width=True,
    )


def dataframe_to_excel_bytes(dataframe: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


def render_classification_table(classification_df: pd.DataFrame) -> None:
    st.subheader("评论主题分类结果")
    st.dataframe(classification_df, use_container_width=True, hide_index=True)
    download_cols = st.columns(2)
    download_cols[0].download_button(
        "下载分类结果 CSV",
        classification_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="comment_topic_classification.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_cols[1].download_button(
        "下载分类结果 Excel",
        dataframe_to_excel_bytes(classification_df, "分类结果"),
        file_name="comment_topic_classification.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


def render_downloads(topic_df: pd.DataFrame, dist_df: pd.DataFrame, classification_df: pd.DataFrame | None = None) -> None:
    st.subheader("下载结果")
    download_cols = st.columns(3 if classification_df is not None else 2)
    download_cols[0].download_button(
        "下载主题关键词 CSV",
        topic_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="lda_topics.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if classification_df is not None:
        download_cols[2].download_button(
            "下载分类结果 CSV",
            classification_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="comment_topic_classification.csv",
            mime="text/csv",
            use_container_width=True,
        )
    download_cols[1].download_button(
        "下载文档主题分布 CSV",
        dist_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="lda_document_topics.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    configure_page()
    st.title(APP_TITLE)
    st.caption(f"快速探索一组文本中的潜在主题、关键词和文档归属。版本：{APP_VERSION}")

    documents = load_documents()
    settings = render_training_controls()

    if len(documents) < 2:
        st.info("请至少提供 2 篇文档。每一行会被视为一篇文档。")
        return

    if settings["min_df"] > len(documents):
        st.error("“词语最少出现文档数”不能大于当前文档数。")
        return

    if st.session_state.get("train_model"):
        try:
            with st.spinner("正在读取文本并加载模型..."):
                train_current_model(documents, settings, settings["n_topics"], 15)
                st.session_state["active_view"] = "words"
        except ValueError as exc:
            st.error(f"模型训练失败：{exc}")
            st.stop()
        st.success("模型已加载完成，可以继续生成权重词、气泡图或强度变化。")

    render_tool_panels(documents, settings)


if __name__ == "__main__":
    main()
