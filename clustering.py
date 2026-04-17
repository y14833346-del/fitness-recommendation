"""
聚类分析模块
实现K-means聚类与群体特征提取
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import hdbscan
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from recommendation import call_minimax_api


# 国家学生体质健康标准等级划分阈值
RELATIVE_LEVEL_THRESHOLDS = {
    "很差": (0, 0.80),
    "较差": (0.80, 0.90),
    "中等": (0.90, 1.10),
    "良好": (1.10, float('inf'))
}


def get_relative_level(value, baseline):
    """根据相对水平判断等级"""
    if baseline == 0:
        return "中等"
    ratio = value / baseline
    for level, (lower, upper) in RELATIVE_LEVEL_THRESHOLDS.items():
        if lower <= ratio < upper:
            return level
    return "中等"


def prepare_clustering_data(df, validation_result, indicator_cols):
    """
    准备聚类数据
    只选择数值型标准化列
    """
    gender_col = validation_result["gender_column"]

    # 找出标准化后的列
    norm_columns = []
    for indicator, original_col in indicator_cols.items():
        norm_col = original_col + '_norm'
        if norm_col in df.columns:
            norm_columns.append(norm_col)

    if not norm_columns:
        # 如果没有标准化列，使用原始列
        for indicator, col in indicator_cols.items():
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                norm_columns.append(col)

    # 提取聚类特征矩阵
    X = df[norm_columns].fillna(0).values

    return X, norm_columns


def find_optimal_clusters(X, min_clusters=10, max_clusters=20, df=None, gender_col=None):
    """
    分层强制配对聚类策略

    1. 男生、女生分别聚类（各自多聚类，得到细粒度子群体）
    2. 计算每个子群体的能力特征
    3. 贪婪配对：将每个男生子群体与最相似的女生子群体合并
    4. 每个合并后的群体都有男有女

    这样得到更多、更精细的混合性别群体
    """
    if df is None or gender_col is None:
        return find_optimal_clusters_hdbscan(X, min_clusters, max_clusters)

    # 获取性别掩码
    def get_gender_mask(data_col, gender_filter):
        if gender_filter == "男":
            mask_text = data_col.astype(str).str.contains("男|boy|male", case=False, na=False)
            mask_num = pd.to_numeric(data_col, errors='coerce') == 1
            return mask_text | mask_num
        else:
            mask_text = data_col.astype(str).str.contains("女|girl|female", case=False, na=False)
            mask_num = pd.to_numeric(data_col, errors='coerce') == 2
            return mask_text | mask_num

    male_mask = get_gender_mask(df[gender_col], "男")
    female_mask = get_gender_mask(df[gender_col], "女")

    print(f"[强制配对聚类] 男性: {male_mask.sum()}, 女性: {female_mask.sum()}")

    # 第一步：男生多聚类（目标15-20个）
    male_k = min(20, male_mask.sum() // 30)  # 每组至少30人
    male_k = max(male_k, 10)
    female_k = min(25, female_mask.sum() // 30)  # 每组至少30人
    female_k = max(female_k, 12)

    print(f"[强制配对聚类] 男生目标{male_k}个群体，女生目标{female_k}个群体")

    # 男生聚类
    X_male = X[male_mask]
    male_labels_sub = gmm_clustering_with_aic(X_male, min_clusters=min(3, male_k), max_clusters=male_k)
    male_n = len(set(male_labels_sub))
    print(f"[强制配对聚类] 男生实际{male_n}个群体")

    # 女生聚类
    X_female = X[female_mask]
    female_labels_sub = gmm_clustering_with_aic(X_female, min_clusters=min(3, female_k), max_clusters=female_k)
    female_n = len(set(female_labels_sub))
    print(f"[强制配对聚类] 女生实际{female_n}个群体")

    # 第二步：计算每个子群体的能力向量
    male_ability = compute_ability_vectors(X_male, male_labels_sub, male_n, is_male=True)
    female_ability = compute_ability_vectors(X_female, female_labels_sub, female_n, is_male=False)

    # 第三步：贪婪配对合并
    labels, n_merged = greedy_pairing_merge(
        X, male_labels_sub, female_labels_sub, male_ability, female_ability, male_mask, female_mask
    )

    print(f"[强制配对聚类] 最终{n_merged}个混合群体")

    # 生成聚类分析图
    elbow_plot = generate_cluster_analysis_plot(X, labels)

    return labels, elbow_plot, n_merged


def compute_ability_vectors(X_sub, labels_sub, n_clusters, is_male):
    """
    计算每个子群体的能力向量
    注意：X_sub 的列顺序假设为：
    [肺活量, 50米跑, 立定跳远, 坐位体前屈, 耐力跑, 力量]
    其中耐力跑和力量的位置在男女之间可能不同（取决于原始数据列顺序）
    """
    ability = {}
    for c in range(n_clusters):
        mask = labels_sub == c
        if mask.sum() == 0:
            continue
        cluster_data = X_sub[mask]

        # 能力向量: [心肺, 速度, 爆发, 柔韧, 力量, 耐力]
        # 注意：对于男生，力量是引体向上(索引5)，耐力是1000米(索引4)
        # 对于女生，力量是仰卧起坐(索引4)，耐力是800米(索引5)
        if is_male:
            # 男生的列顺序：肺活量,50米,立定跳远,体前屈,1000米,引体向上
            abilities = [
                cluster_data[:, 0].mean(),  # 肺活量
                1 - cluster_data[:, 1].mean(),  # 50米 (取反)
                cluster_data[:, 2].mean(),  # 立定跳远
                cluster_data[:, 3].mean(),  # 坐位体前屈
                cluster_data[:, 5].mean(),  # 力量(引体向上)
                1 - cluster_data[:, 4].mean(),  # 耐力(1000米，取反)
            ]
        else:
            # 女生的列顺序：肺活量,50米,立定跳远,体前屈,仰卧起坐,800米
            abilities = [
                cluster_data[:, 0].mean(),  # 肺活量
                1 - cluster_data[:, 1].mean(),  # 50米 (取反)
                cluster_data[:, 2].mean(),  # 立定跳远
                cluster_data[:, 3].mean(),  # 坐位体前屈
                cluster_data[:, 4].mean(),  # 力量(仰卧起坐)
                1 - cluster_data[:, 5].mean(),  # 耐力(800米，取反)
            ]
        ability[c] = {
            "size": mask.sum(),
            "abilities": abilities
        }
    return ability


def greedy_pairing_merge(X, male_labels_sub, female_labels_sub, male_ability, female_ability, male_mask, female_mask):
    """
    贪婪配对合并策略
    将每个男生群体与最相似的女生群体配对合并
    """
    all_labels = np.full(len(X), -1, dtype=int)
    used_female = set()
    merged_id = 0

    # 按群体大小排序男生
    male_ids = sorted(male_ability.keys(), key=lambda x: male_ability[x]["size"], reverse=True)

    for male_id in male_ids:
        m_abilities = male_ability[male_id]["abilities"]

        # 找最相似的未配对女生
        best_sim = -999
        best_female_id = None
        for female_id in female_ability.keys():
            if female_id in used_female:
                continue
            f_abilities = female_ability[female_id]["abilities"]
            sim = cosine_similarity(m_abilities, f_abilities)
            if sim > best_sim:
                best_sim = sim
                best_female_id = female_id

        # 构建掩码
        male_indices = np.where(male_mask)[0][np.where(male_labels_sub == male_id)[0]]
        male_cluster_mask = np.zeros(len(X), dtype=bool)
        male_cluster_mask[male_indices] = True

        if best_female_id is not None:
            used_female.add(best_female_id)
            female_indices = np.where(female_mask)[0][np.where(female_labels_sub == best_female_id)[0]]
            female_cluster_mask = np.zeros(len(X), dtype=bool)
            female_cluster_mask[female_indices] = True

            combined = male_cluster_mask | female_cluster_mask
            all_labels[combined] = merged_id
            print(f"[配对] 群体{merged_id+1}: 男{male_ability[male_id]['size']} + 女{female_ability[best_female_id]['size']} = {combined.sum()}, 相似度={best_sim:.3f}")
        else:
            # 没有可配对的女生了
            all_labels[male_cluster_mask] = merged_id
            print(f"[配对] 群体{merged_id+1}: 男{male_ability[male_id]['size']}单独")

        merged_id += 1

    # 处理剩余女生：强制分配到已有群体中（分散到最大的几个群体）
    remaining_female = []
    for female_id in female_ability.keys():
        if female_id not in used_female:
            female_indices = np.where(female_mask)[0][np.where(female_labels_sub == female_id)[0]]
            remaining_female.extend(female_indices)

    if remaining_female:
        print(f"[配对] 剩余{len(remaining_female)}名女生分配到各群体")
        # 按当前群体大小排序，将剩余女生分配到最大的群体
        current_sizes = []
        for mid in range(merged_id):
            current_sizes.append((mid, np.sum(all_labels == mid)))
        current_sizes.sort(key=lambda x: x[1], reverse=True)

        # 分散分配
        for i, fem_idx in enumerate(remaining_female):
            target_cluster = current_sizes[i % merged_id][0]
            all_labels[fem_idx] = target_cluster

    # 重新计算最终群体数（去除空群体）
    unique_labels = sorted(set(all_labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    final_labels = np.array([label_map[l] for l in all_labels])

    print(f"[配对] 最终{len(unique_labels)}个有效群体")

    return final_labels, len(unique_labels)


def gmm_clustering_with_aic(X, min_clusters=3, max_clusters=10):
    """
    使用高斯混合模型(GMM)+AIC/BIC自动确定最优聚类数
    """
    lowest_aic = np.inf
    best_labels = None
    best_n = min_clusters

    K_range = range(min_clusters, max_clusters + 1)

    for n in K_range:
        gmm = GaussianMixture(
            n_components=n,
            covariance_type='full',
            random_state=42,
            n_init=3
        )
        gmm.fit(X)
        aic = gmm.aic(X)

        if aic < lowest_aic:
            lowest_aic = aic
            best_n = n
            best_labels = gmm.predict(X)

    print(f"[GMM] 最优聚类数: {best_n} (AIC={lowest_aic:.2f})")
    return best_labels
    """
    基于能力相似性匹配男女群体并合并
    返回全局标签和合并后的群体数
    """
    # 计算每个群体的能力向量（使用相对水平）
    # 能力维度：心肺耐力、力量、爆发力、速度、柔韧性

    # 为每个群体计算能力得分
    male_ability = compute_ability_scores(male_labels, X[male_mask], is_male=True)
    female_ability = compute_ability_scores(female_labels, X[female_mask], is_male=False)

    print(f"[匹配] 男生群体能力: {len(male_ability)}")
    print(f"[匹配] 女生群体能力: {len(female_ability)}")

    # 构建男女群体的配对关系
    # 策略：按能力相似度贪婪匹配
    merged_id = 0
    all_labels = np.full(len(X), -1, dtype=int)

    # 获取排序后的群体列表
    male_ids = sorted(male_ability.keys(), key=lambda x: male_ability[x]["size"], reverse=True)
    female_ids = sorted(female_ability.keys(), key=lambda x: female_ability[x]["size"], reverse=True)

    used_female = set()

    # 贪心匹配：每个男生群体找最相似的女生群体
    for male_id in male_ids:
        m_data = male_ability[male_id]

        best_sim = -999
        best_female_id = None

        for female_id in female_ids:
            if female_id in used_female:
                continue
            f_data = female_ability[female_id]
            sim = cosine_similarity(m_data["abilities"], f_data["abilities"])
            if sim > best_sim:
                best_sim = sim
                best_female_id = female_id

        # 构建男生群体的掩码
        male_cluster_mask = np.zeros(len(X), dtype=bool)
        male_indices_in_original = np.where(male_mask)[0]
        male_cluster_indices = np.where(male_labels == male_id)[0]
        male_cluster_mask[male_indices_in_original[male_cluster_indices]] = True

        if best_female_id is not None and best_sim > 0.3:
            # 配对成功
            used_female.add(best_female_id)

            # 构建女生群体的掩码
            female_cluster_mask = np.zeros(len(X), dtype=bool)
            female_indices_in_original = np.where(female_mask)[0]
            female_cluster_indices = np.where(female_labels == best_female_id)[0]
            female_cluster_mask[female_indices_in_original[female_cluster_indices]] = True

            combined_mask = male_cluster_mask | female_cluster_mask
            all_labels[combined_mask] = merged_id
            print(f"[匹配] 群体{merged_id+1}: M{male_id}({male_ability[male_id]['size']}人)+F{best_female_id}({female_ability[best_female_id]['size']}人), 相似度={best_sim:.3f}")
            merged_id += 1
        else:
            # 没有匹配到合适的女生，单独成组
            all_labels[male_cluster_mask] = merged_id
            print(f"[匹配] 群体{merged_id+1}: M{male_id}({male_ability[male_id]['size']}人)单独")
            merged_id += 1

    # 处理剩余女生
    for female_id in female_ids:
        if female_id not in used_female:
            female_cluster_mask = np.zeros(len(X), dtype=bool)
            female_indices_in_original = np.where(female_mask)[0]
            female_cluster_indices = np.where(female_labels == female_id)[0]
            female_cluster_mask[female_indices_in_original[female_cluster_indices]] = True
            all_labels[female_cluster_mask] = merged_id
            print(f"[匹配] 群体{merged_id+1}: F{female_id}({female_ability[female_id]['size']}人)单独")
            merged_id += 1

    print(f"[匹配] 最终合并为 {merged_id} 个推荐群体")
    return all_labels, merged_id


def compute_ability_scores(labels, X, is_male):
    """
    计算每个群体的能力得分向量
    能力维度: [心肺耐力, 力量, 爆发力, 速度, 柔韧性]
    """
    # 列索引映射（假设按顺序）
    # Vital capacity, 50-m dash, Standing long jump, Sit and reach, endurance, sit-ups/pull-ups
    n_cols = X.shape[1]

    ability_scores = {}
    for c in range(len(set(labels))):
        mask = labels == c
        if mask.sum() == 0:
            continue

        cluster_data = X[mask]
        means = cluster_data.mean(axis=0)

        # 能力向量（归一化到0-1）
        abilities = []

        # 心肺耐力：肺活量 (索引0)
        vitality = means[0] if n_cols > 0 else 0.5

        # 速度：50米跑 (索引1) - 值越小越好，取反
        speed = 1 - means[1] if n_cols > 1 else 0.5

        # 爆发力：立定跳远 (索引2)
        jump = means[2] if n_cols > 2 else 0.5

        # 柔韧性：坐位体前屈 (索引3)
        flexibility = means[3] if n_cols > 3 else 0.5

        # 力量：引体向上(男)/仰卧起坐(女) - 取最后一个或倒数第二个指标
        if is_male:
            strength = means[-1] if n_cols > 5 else 0.5  # Pull-ups
        else:
            strength = means[-2] if n_cols > 5 else 0.5  # sit-ups

        abilities = [vitality, speed, jump, flexibility, strength]

        # 归一化
        max_val = max(abilities) if max(abilities) > 0 else 1
        abilities = [a / max_val for a in abilities]

        ability_scores[c] = {
            "size": mask.sum(),
            "abilities": abilities
        }

    return ability_scores


def cosine_similarity(a, b):
    """计算余弦相似度"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a * norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)


def find_optimal_clusters_hdbscan(X, min_clusters=3, max_clusters=10):
    """
    使用HDBSCAN密度聚类自动确定最优聚类数（标准版本）
    """
    # 先尝试HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50,  # 最小聚类大小
        min_samples=10,       # 核心点数量
        cluster_selection_epsilon=0.1,  # 聚类选择epsilon
        metric='euclidean'
    )
    labels = clusterer.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # 排除噪声点
    n_noise = (labels == -1).sum()

    print(f"[HDBSCAN] 发现 {n_clusters} 个聚类, {n_noise} 个噪声点")

    # 如果聚类数太少或太多，使用GMM
    if n_clusters < min_clusters or n_clusters > max_clusters:
        print(f"[HDBSCAN] 聚类数({n_clusters})不在合理范围[{min_clusters},{max_clusters}]，改用GMM+AIC")
        labels = gmm_clustering_with_aic(X, min_clusters, max_clusters)
        n_clusters = len(set(labels))
        print(f"[GMM] 最终聚类数: {n_clusters}")

    # 生成聚类分析图
    elbow_plot = generate_cluster_analysis_plot(X, labels)

    return labels, elbow_plot, n_clusters


def cluster_within_group(X, min_clusters=2, max_clusters=6):
    """
    对分组数据进行聚类
    优先使用HDBSCAN，如果聚类数不合适则用GMM
    """
    # 尝试HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(20, len(X) // 50),  # 动态调整最小聚类大小
        min_samples=5,
        cluster_selection_epsilon=0.05,
        metric='euclidean'
    )
    labels = clusterer.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters < min_clusters or n_clusters > max_clusters:
        # 使用GMM
        labels = gmm_clustering_with_aic(X, min_clusters, max_clusters)
        n_clusters = len(set(labels))

    return labels, n_clusters


def gmm_clustering_with_aic(X, min_clusters=3, max_clusters=10):
    """
    使用高斯混合模型(GMM)+AIC/BIC自动确定最优聚类数
    GMM是软聚类，每个点有属于各聚类的概率
    """
    lowest_aic = np.inf
    best_labels = None
    best_n = min_clusters

    K_range = range(min_clusters, max_clusters + 1)

    for n in K_range:
        gmm = GaussianMixture(
            n_components=n,
            covariance_type='full',
            random_state=42,
            n_init=3
        )
        gmm.fit(X)
        aic = gmm.aic(X)

        if aic < lowest_aic:
            lowest_aic = aic
            best_n = n
            best_labels = gmm.predict(X)

    print(f"[GMM] 最优聚类数: {best_n} (AIC={lowest_aic:.2f})")
    return best_labels


def generate_cluster_analysis_plot(X, labels):
    """
    生成聚类分析可视化图
    使用PCA降维到2D进行可视化
    """
    from sklearn.decomposition import PCA

    # PCA降维到2D用于可视化
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    n_clusters = len(set(labels))

    # 左图：聚类散点图
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 10)))

    for i, label in enumerate(sorted(set(labels))):
        mask = labels == label
        if label == -1:
            # 噪声点用灰色
            axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                          c='gray', alpha=0.3, s=10, label='噪声')
        else:
            axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                          c=[colors[i % len(colors)]], alpha=0.6, s=15,
                          label=f'群体{label+1}')

    axes[0].set_xlabel('主成分1')
    axes[0].set_ylabel('主成分2')
    axes[0].set_title(f'聚类分布 (共{n_clusters}个群体)')
    axes[0].legend(loc='best', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 右图：各聚类样本数量
    cluster_counts = [np.sum(labels == i) for i in sorted(set(labels)) if i != -1]
    cluster_labels = [f'群体{i+1}' for i in range(len(cluster_counts))]

    bars = axes[1].bar(cluster_labels, cluster_counts, color=colors[:len(cluster_counts)])
    axes[1].set_xlabel('聚类')
    axes[1].set_ylabel('样本数量')
    axes[1].set_title('各聚类样本分布')

    # 在柱状图上显示数量
    for bar, count in zip(bars, cluster_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                     str(count), ha='center', va='bottom', fontsize=9)

    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 转换为base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_base64


def perform_clustering(X, n_clusters):
    """
    执行K-means聚类（保留作为备选）
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    return {
        "labels": labels,
        "centers": centers,
        "inertia": kmeans.inertia_
    }


def extract_cluster_features(df, labels, validation_result, indicator_cols):
    """
    提取群体特征
    为每个聚类准备标准化后的能力向量数据，供LLM分析

    关键：使用和聚类时相同的标准化逻辑，确保数据一致性
    能力向量：[心肺, 速度, 爆发, 柔韧, 力量, 耐力]
    """
    gender_col = validation_result["gender_column"]

    # 获取标准化列（_norm后缀）
    norm_columns = {}
    for indicator, col in indicator_cols.items():
        norm_col = col + '_norm'
        if norm_col in df.columns:
            norm_columns[indicator] = norm_col

    # 性别掩码辅助函数
    def get_gender_mask(data_col, gender_filter):
        if gender_filter == "男":
            mask_text = data_col.astype(str).str.contains("男|boy|male", case=False, na=False)
            mask_num = pd.to_numeric(data_col, errors='coerce') == 1
            return mask_text | mask_num
        else:
            mask_text = data_col.astype(str).str.contains("女|girl|female", case=False, na=False)
            mask_num = pd.to_numeric(data_col, errors='coerce') == 2
            return mask_text | mask_num

    def get_dominant_gender(data_col):
        male_mask = get_gender_mask(data_col, "男")
        female_mask = get_gender_mask(data_col, "女")
        male_count = male_mask.sum()
        female_count = female_mask.sum()
        if male_count > female_count:
            return "男生主导" if male_count > female_count * 1.5 else "男生较多"
        elif female_count > male_count:
            return "女生主导" if female_count > male_count * 1.5 else "女生较多"
        return "均衡"

    # 检查标准化列是否存在
    if not norm_columns:
        # 如果没有标准化列，无法计算，返回空
        return {}, {}

    # 构建标准化数据矩阵 - 动态适配实际列
    # 核心4列（男女都有）：肺活量, 50米, 立定跳远, 坐位体前屈
    # 加上力量（引体向上/仰卧起坐）和耐力（1000米/800米）
    norm_cols_ordered = []
    ability_map = []  # [(能力名, 是否取反), ...]

    # 基础4项（男女通用）
    for indicator in ["肺活量", "50米跑", "立定跳远", "坐位体前屈"]:
        if indicator in norm_columns:
            norm_cols_ordered.append(norm_columns[indicator])
            name_map = {"肺活量": "心肺", "50米跑": "速度", "立定跳远": "爆发", "坐位体前屈": "柔韧"}
            ability_map.append((name_map.get(indicator, indicator), indicator == "50米跑"))

    # 力量列：引体向上(男)或仰卧起坐(女)
    has_male_strength = "引体向上" in norm_columns
    has_female_strength = "仰卧起坐" in norm_columns
    if has_male_strength:
        norm_cols_ordered.append(norm_columns["引体向上"])
        ability_map.append(("力量", False))
    elif has_female_strength:
        norm_cols_ordered.append(norm_columns["仰卧起坐"])
        ability_map.append(("力量", False))

    # 耐力列：耐力跑
    has_endurance = "耐力跑" in norm_columns
    if has_endurance:
        norm_cols_ordered.append(norm_columns["耐力跑"])
        ability_map.append(("耐力", True))

    n_abilities = len(ability_map)
    print(f"[extract_cluster_features] 标准化列数: {len(norm_cols_ordered)}, 能力数: {n_abilities}")
    print(f"[extract_cluster_features] 列: {norm_cols_ordered}")
    print(f"[extract_cluster_features] 能力映射: {ability_map}")

    # 构建X_all
    X_all = df[norm_cols_ordered].fillna(0.5).values if norm_cols_ordered else np.array([]).reshape(0, 0)

    def calc_ability_vector(X_subset):
        """从标准化数据计算能力向量，使用动态列布局"""
        if len(X_subset) == 0 or n_abilities == 0:
            return None
        abilities = []
        for i, (name, inverse) in enumerate(ability_map):
            val = X_subset[:, i].mean()
            if inverse:
                val = 1 - val
            abilities.append(val)
        return abilities

    # 计算baseline
    male_mask = get_gender_mask(df[gender_col], "男")
    female_mask = get_gender_mask(df[gender_col], "女")

    male_ability_baseline = calc_ability_vector(X_all[male_mask]) if male_mask.sum() > 0 else None
    female_ability_baseline = calc_ability_vector(X_all[female_mask]) if female_mask.sum() > 0 else None

    # 能力名称列表
    ability_names = [name for name, _ in ability_map]

    # 为每个聚类生成特征
    cluster_features = {}
    valid_labels = [l for l in set(labels) if l != -1]

    for cluster_id in valid_labels:
        mask = labels == cluster_id
        cluster_size = mask.sum()
        cluster_ratio = cluster_size / len(labels) * 100

        dominant_gender = get_dominant_gender(df.loc[mask, gender_col])

        # 分别获取男女 sub-group 的 mask
        male_sub_mask = mask & male_mask if 'male_mask' in dir() else np.zeros(len(df), dtype=bool)
        female_sub_mask = mask & female_mask if 'female_mask' in dir() else np.zeros(len(df), dtype=bool)

        X_male_sub = X_all[male_sub_mask] if male_sub_mask.sum() > 0 else np.array([]).reshape(0, n_abilities)
        X_female_sub = X_all[female_sub_mask] if female_sub_mask.sum() > 0 else np.array([]).reshape(0, n_abilities)

        # 计算各自的能力向量
        male_ability_vec = calc_ability_vector(X_male_sub) if male_sub_mask.sum() > 0 else None
        female_ability_vec = calc_ability_vector(X_female_sub) if female_sub_mask.sum() > 0 else None

        # 混合 cluster：分别用各自性别基准计算相对水平，然后合并
        # 纯单一性别 cluster：直接用该性别的基准
        is_male_only = female_sub_mask.sum() == 0
        is_female_only = male_sub_mask.sum() == 0

        relative_data = {}
        if is_male_only and male_ability_vec and male_ability_baseline:
            # 纯男生 cluster
            for i, ability_name in enumerate(ability_names):
                key = ability_name  # 不带性别后缀
                relative_data[key] = {
                    "mean": male_ability_vec[i],
                    "relative": male_ability_vec[i] / male_ability_baseline[i] if male_ability_baseline[i] != 0 else 1.0
                }
        elif is_female_only and female_ability_vec and female_ability_baseline:
            # 纯女生 cluster
            for i, ability_name in enumerate(ability_names):
                key = ability_name  # 不带性别后缀
                relative_data[key] = {
                    "mean": female_ability_vec[i],
                    "relative": female_ability_vec[i] / female_ability_baseline[i] if female_ability_baseline[i] != 0 else 1.0
                }
        elif male_ability_vec and female_ability_vec:
            # 混合 cluster：男女分别计算相对水平，然后取平均
            # 因为配对时是基于相似的 ability profile，所以取平均是合理的
            for i, ability_name in enumerate(ability_names):
                key = ability_name  # 不带性别后缀
                male_rel = male_ability_vec[i] / male_ability_baseline[i] if male_ability_baseline and male_ability_baseline[i] != 0 else 1.0
                female_rel = female_ability_vec[i] / female_ability_baseline[i] if female_ability_baseline and female_ability_baseline[i] != 0 else 1.0
                # 取平均（因为配对的两个 sub-group 能力相似）
                avg_relative = (male_rel + female_rel) / 2
                avg_mean = (male_ability_vec[i] + female_ability_vec[i]) / 2
                relative_data[key] = {
                    "mean": avg_mean,
                    "relative": avg_relative
                }

        cluster_features[cluster_id] = {
            "cluster_id": cluster_id,
            "label": f"C{cluster_id + 1}",
            "size": int(cluster_size),
            "ratio": round(cluster_ratio, 1),
            "dominant_gender": dominant_gender,
            "indicator_data": {},
            "relative_data": relative_data,
            "cluster_type": None,
            "description": None,
            "weak_features": [],
            "strong_features": []
        }

    return cluster_features, {}


def analyze_clusters_with_llm(cluster_features, baselines, api_key):
    """
    使用LLM批量分析所有聚类，生成群体类型、描述、优劣势
    一次调用完成所有聚类分析，传入相对水平数据

    Args:
        cluster_features: 聚类特征字典
        baselines: 全校各指标基准值字典
        api_key: API密钥

    Returns:
        更新后的聚类特征字典
    """
    if not api_key:
        return cluster_features

    try:
        from recommendation import analyze_all_clusters_with_llm

        # 构建批量分析数据（使用相对水平）
        batch_data = {}
        for cluster_id, info in cluster_features.items():
            batch_data[cluster_id] = {
                "label": info.get("label", f"C{cluster_id+1}"),
                "dominant_gender": info.get("dominant_gender", "未知"),
                "relative_data": info.get("relative_data", {})
            }

        print(f"[LLM批量分析] 开始分析 {len(batch_data)} 个聚类")

        # 一次调用分析所有聚类
        results = analyze_all_clusters_with_llm(batch_data, baselines, api_key)

        if results:
            for cluster_id, result in results.items():
                if cluster_id in cluster_features:
                    info = cluster_features[cluster_id]
                    info["cluster_type"] = result.get("cluster_type", "待评估")
                    info["description"] = result.get("description", "")
                    info["weak_features"] = result.get("weak_features", [])
                    info["strong_features"] = result.get("strong_features", [])
            print(f"[LLM批量分析] 完成 {len(results)} 个聚类分析")
        else:
            print(f"[LLM批量分析] 失败")

    except Exception as e:
        print(f"LLM批量分析失败: {e}")
        import traceback
        traceback.print_exc()

    return cluster_features


def generate_cluster_type(weak_features, strong_features, dominant_gender):
    """
    根据薄弱项和优势项生成群体类型标签
    """
    # 统计各类能力
    ability_counts = {"力量": 0, "心肺耐力": 0, "柔韧性": 0, "爆发力": 0, "速度": 0}

    indicator_ability_map = {
        "肺活量": "心肺耐力",
        "1000米跑": "心肺耐力",
        "800米跑": "心肺耐力",
        "引体向上": "力量",
        "仰卧起坐": "力量",
        "力量": "力量",  # 统一力量指标（男女）
        "立定跳远": "爆发力",
        "50米跑": "速度",
        "坐位体前屈": "柔韧性"
    }

    for wf in weak_features:
        ability = indicator_ability_map.get(wf)
        if ability:
            ability_counts[ability] -= 1

    for sf in strong_features:
        ability = indicator_ability_map.get(sf)
        if ability:
            ability_counts[ability] += 1

    # 找出最需要提升的能力
    min_ability = min(ability_counts, key=ability_counts.get)
    min_count = ability_counts[min_ability]

    # 找出最强项
    max_ability = max(ability_counts, key=ability_counts.get)
    max_count = ability_counts[max_ability]

    # 生成类型标签
    if min_count < 0 and max_count > 0:
        return f"{max_ability}型"
    elif min_count < 0:
        return f"{min_ability}薄弱型"
    elif max_count > 0:
        return f"{max_ability}型"
    else:
        return "均衡型"


def generate_cluster_description(cluster_type, weak_features, strong_features, dominant_gender):
    """
    生成群体描述文字
    """
    parts = []

    # 性别信息
    gender_text = "男生" if "男" in str(dominant_gender) else "女生" if "女" in str(dominant_gender) else ""
    if gender_text:
        parts.append(f"该群体以{gender_text}为主")

    # 类型信息
    parts.append(f"整体属于「{cluster_type}」")

    # 薄弱项描述
    if weak_features:
        weak_text = "、".join(weak_features[:3])
        if len(weak_features) > 3:
            weak_text += f"等{len(weak_features)}项"
        parts.append(f"薄弱项: {weak_text}")

    # 优势项描述
    if strong_features:
        strong_text = "、".join(strong_features[:3])
        if len(strong_features) > 3:
            strong_text += f"等{len(strong_features)}项"
        parts.append(f"优势项: {strong_text}")

    return " | ".join(parts)


def optimize_features_with_ai(cluster_features, api_key=None):
    """
    AI辅助优化特征（可选步骤）
    将量化特征转化为教学友好的语言
    """
    if not api_key:
        return cluster_features

    try:
        # 构建输入
        feature_summary = []
        for cid, info in cluster_features.items():
            ratio = info["ratio"]
            features = ", ".join(info["features"])
            feature_summary.append(f"C{cid+1}: {ratio}% {features}")

        prompt = f"""以下是体育课程推荐系统的学生聚类结果，每行是一个群体：
{chr(10).join(feature_summary)}

请将上述量化特征转化为教学语言，要求：
1. 保留核心信息（如"肺活量很差"）
2. 转化为学生和家长能理解的语言
3. 每类一句话概括
4. 格式：群体编号: 概括语

示例：
C1: 心肺耐力薄弱群体（32%），建议重点加强有氧运动
C2: 柔韧性较差群体（18%），建议增加拉伸类课程"""

        # 调用MiniMax API
        response = call_minimax_api(prompt, api_key)
        if not response:
            return cluster_features

        # 解析AI输出
        ai_descriptions = {}
        for line in response.split('\n'):
            if ':' in line and line[0] == 'C':
                parts = line.split(':', 1)
                cid = int(parts[0][1]) - 1
                ai_descriptions[cid] = parts[1].strip()

        # 更新聚类特征
        for cid in cluster_features:
            if cid in ai_descriptions:
                cluster_features[cid]["ai_description"] = ai_descriptions[cid]

    except Exception as e:
        print(f"AI优化失败: {e}")

    return cluster_features


def clustering_pipeline(df, validation_result, indicator_cols, api_key=None):
    """
    完整聚类流程
    使用HDBSCAN密度聚类自动确定最优聚类数
    """
    print("=" * 50)
    print("开始聚类分析")
    print("=" * 50)

    # 准备数据
    X, norm_columns = prepare_clustering_data(df, validation_result, indicator_cols)
    print(f"聚类特征矩阵: {X.shape}")

    # 使用HDBSCAN自动确定最优聚类数
    labels, elbow_plot, n_clusters = find_optimal_clusters(X, min_clusters=3, max_clusters=8)
    print(f"自动确定聚类数: {n_clusters}")

    # 提取群体特征（原始数据）
    cluster_features, baselines = extract_cluster_features(
        df, labels, validation_result, indicator_cols
    )

    # 使用LLM分析（必须）
    if api_key:
        print("正在调用LLM分析群体特征...")
        cluster_features = analyze_clusters_with_llm(cluster_features, baselines, api_key)
    else:
        print("警告: 未配置API Key，无法进行LLM分析")

    print("=" * 50)
    print("聚类分析完成")
    print("=" * 50)

    return {
        "optimal_k": n_clusters,
        "labels": labels.tolist(),
        "cluster_features": cluster_features,
        "baselines": baselines,
        "elbow_plot": elbow_plot,
        "norm_columns": norm_columns
    }


if __name__ == "__main__":
    # 测试用
    import os
    os.chdir("/Users/bytedance/Desktop/毕设")
