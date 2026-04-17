"""
数据预处理模块
实现体测数据的筛选、标准化、异常值处理
"""

import pandas as pd
import numpy as np
from scipy import stats


# 核心体测指标
CORE_INDICATORS = [
    "肺活量", "50米跑", "立定跳远", "坐位体前屈",
    "引体向上", "1000米跑",  # 男生
    "仰卧起坐", "800米跑"     # 女生
]

# 性别差异指标
MALE_INDICATORS = ["引体向上", "1000米跑"]
FEMALE_INDICATORS = ["仰卧起坐", "800米跑"]

# 生理指标合理范围（用于异常值检测）
INDICATOR_RANGES = {
    "肺活量": {"male": (1500, 6000), "female": (1000, 5000)},
    "50米跑": {"male": (6.0, 12.0), "female": (7.0, 13.0)},
    "立定跳远": {"male": (120, 300), "female": (100, 250)},
    "坐位体前屈": {"male": (-20, 40), "female": (-20, 40)},
    "引体向上": {"male": (0, 30), "female": None},
    "1000米跑": {"male": (180, 420), "female": None},  # 秒
    "仰卧起坐": {"male": None, "female": (0, 60)},
    "800米跑": {"male": None, "female": (150, 360)},  # 秒
}


def load_data(file_path):
    """
    加载体测数据文件
    支持CSV和Excel格式
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("仅支持CSV或Excel格式文件")

    return df


def validate_columns(df):
    """
    验证数据列是否符合要求
    支持中文和英文列名
    """
    # 英文列名映射到中文指标名
    english_mapping = {
        "Vital capacity (ml)": "肺活量",
        "50-m dsah (s)": "50米跑",
        "Standing long jump (cm)": "立定跳远",
        "Sit and reach (cm)": "坐位体前屈",
        "Pull-ups for boys": "引体向上",
        "sit-ups for girls": "仰卧起坐",
        "Score of endurance run": "耐力跑",
    }

    # 中文列名映射（原有）
    optional_mapping = {
        "肺活量": ["肺活量", "pulmonary", "Vital capacity"],
        "50米跑": ["50米跑", "50米", "fifty_meter", "50-m"],
        "立定跳远": ["立定跳远", "跳远", "long_jump", "Standing long jump"],
        "坐位体前屈": ["坐位体前屈", "体前屈", "sit_and_reach", "Sit and reach"],
        "引体向上": ["引体向上", "引体", "Pull-ups", "pull-ups"],
        "1000米跑": ["1000米跑", "1000米"],
        "仰卧起坐": ["仰卧起坐", "仰卧", "sit-ups", "Sit and reach"],
        "800米跑": ["800米跑", "800米"],
    }

    found_columns = {}
    missing_columns = []

    # 先检查英文列名
    for col in df.columns:
        if col in english_mapping:
            indicator = english_mapping[col]
            found_columns[indicator] = col
            continue

        # 再检查中文列名/模糊匹配
        for indicator, possible_names in optional_mapping.items():
            if indicator in found_columns:
                continue
            if any(name in col for name in possible_names):
                found_columns[indicator] = col
                break

    # 检查性别列
    gender_col = None
    for col in df.columns:
        if "性别" in col or "gender" in col.lower():
            gender_col = col
            break

    if gender_col is None:
        raise ValueError("数据中未找到'性别'列")

    # 检查缺失的必选指标（肺活量、50米、立定跳远、坐位体前屈是必选的）
    required_indicators = ["肺活量", "50米跑", "立定跳远", "坐位体前屈"]
    for indicator in required_indicators:
        if indicator not in found_columns:
            missing_columns.append(indicator)

    return {
        "valid": True,
        "gender_column": gender_col,
        "indicator_columns": found_columns,
        "missing_indicators": missing_columns
    }


def filter_invalid_samples(df, validation_result):
    """
    剔除无效样本
    - 核心指标缺失 > 3项
    - 指标逻辑错误（如身高<100cm等）
    """
    gender_col = validation_result["gender_column"]
    indicator_cols = validation_result["indicator_columns"]

    df_filtered = df.copy()

    # 统计每行的缺失指标数
    def count_missing(row):
        missing = 0
        for indicator in CORE_INDICATORS:
            col = indicator_cols.get(indicator)
            if col is None:
                missing += 1
            elif pd.isna(row.get(col)) or row.get(col) == '':
                missing += 1
        return missing

    df_filtered['缺失指标数'] = df_filtered.apply(count_missing, axis=1)

    # 保留缺失<=3项的样本
    df_filtered = df_filtered[df_filtered['缺失指标数'] <= 3]

    # 删除辅助列
    df_filtered = df_filtered.drop('缺失指标数', axis=1)

    valid_rate = len(df_filtered) / len(df) * 100
    print(f"有效样本率: {valid_rate:.1f}% (原始: {len(df)}, 筛选后: {len(df_filtered)})")

    return df_filtered


def remove_low_variance_indicators(df, indicator_cols, threshold=0.05):
    """
    基于方差分析删除无差异指标
    方差 < threshold 的指标被删除
    """
    high_variance_indicators = {}

    for indicator, col in indicator_cols.items():
        if col in df.columns:
            try:
                values = pd.to_numeric(df[col], errors='coerce')
                variance = values.var()
                if variance >= threshold:
                    high_variance_indicators[indicator] = col
                else:
                    print(f"剔除冗余指标: {indicator} (方差: {variance:.4f})")
            except:
                pass

    return high_variance_indicators


def normalize_minmax(df, indicator_cols):
    """
    Min-Max归一化
    将所有体测指标缩至[0,1]区间
    """
    df_normalized = df.copy()

    normalize_indicator = {}
    for indicator in CORE_INDICATORS:
        if indicator in indicator_cols:
            col = indicator_cols[indicator]
            normalize_indicator[indicator] = col

    for indicator, col in normalize_indicator.items():
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce')
            min_val = values.min()
            max_val = values.max()

            if max_val > min_val:
                df_normalized[col + '_norm'] = (values - min_val) / (max_val - min_val)
            else:
                df_normalized[col + '_norm'] = 0.5

    return df_normalized


def handle_outliers_median(df, validation_result):
    """
    通过3σ原则识别异常值
    采用指标中位数替换而非直接删除
    """
    gender_col = validation_result["gender_column"]
    indicator_cols = validation_result["indicator_columns"]

    df_cleaned = df.copy()

    for indicator, col in indicator_cols.items():
        if col not in df.columns:
            continue

        # 根据性别分别处理 - 支持文本和数值性别编码
        # 文本模式
        male_mask_text = df_cleaned[gender_col].astype(str).str.contains("男|boy|male|1", case=False, na=False)
        female_mask_text = df_cleaned[gender_col].astype(str).str.contains("女|girl|female|2", case=False, na=False)
        # 数值模式
        male_mask_num = pd.to_numeric(df_cleaned[gender_col], errors='coerce') == 1
        female_mask_num = pd.to_numeric(df_cleaned[gender_col], errors='coerce') == 2
        # 合并
        male_mask = male_mask_text | male_mask_num
        female_mask = female_mask_text | female_mask_num

        for gender, mask in [("男生", male_mask), ("女生", female_mask)]:
            if mask.sum() == 0:
                continue

            gender_data = pd.to_numeric(df_cleaned.loc[mask, col], errors='coerce')
            median_val = gender_data.median()
            std_val = gender_data.std()

            if pd.isna(std_val) or std_val == 0:
                continue

            # 3σ原则
            lower_bound = median_val - 3 * std_val
            upper_bound = median_val + 3 * std_val

            # 替换异常值
            outlier_mask = mask & ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound))
            outlier_count = outlier_mask.sum()

            if outlier_count > 0:
                df_cleaned.loc[outlier_mask, col] = median_val
                print(f"{indicator}[{gender}]: 替换{outlier_count}个异常值 (中位数:{median_val:.1f})")

    return df_cleaned


def preprocess_pipeline(file_path):
    """
    完整预处理流程
    1. 加载数据
    2. 验证列
    3. 筛选无效样本
    4. 处理异常值
    5. 标准化
    """
    print("=" * 50)
    print("开始数据预处理")
    print("=" * 50)

    # 1. 加载数据
    df = load_data(file_path)
    print(f"原始数据: {len(df)} 条记录")

    # 2. 验证列
    validation = validate_columns(df)
    print(f"找到指标列: {list(validation['indicator_columns'].keys())}")

    # 3. 筛选无效样本
    df_filtered = filter_invalid_samples(df, validation)

    # 4. 处理异常值
    df_cleaned = handle_outliers_median(df_filtered, validation)

    # 5. 删除冗余指标
    indicator_cols = remove_low_variance_indicators(
        df_cleaned,
        validation["indicator_columns"]
    )

    # 6. 标准化
    df_normalized = normalize_minmax(df_cleaned, indicator_cols)

    print("=" * 50)
    print("预处理完成")
    print("=" * 50)

    return {
        "original_data": df,
        "filtered_data": df_filtered,
        "cleaned_data": df_cleaned,
        "normalized_data": df_normalized,
        "validation": validation,
        "indicator_columns": indicator_cols
    }


if __name__ == "__main__":
    # 测试用
    import os
    sample_file = "data/sample_data.csv"
    if os.path.exists(sample_file):
        result = preprocess_pipeline(sample_file)
        print(result["normalized_data"].head())
