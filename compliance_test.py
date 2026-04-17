"""
API遵循度对比测试
测试MiniMax-M2.7和Doubao在不同Prompt约束下的遵循程度
包含重试机制保证测试完成
"""

import os
import json
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# API配置
MINIMAX_API_KEY = "sk-cp-7w8ntV94GDKBPOSVCVE7Ub_y0DH02qmY-qINgg-RoHoFSbUVlB3NXxoFMRlCJ4OC6AD6I7EiZVNUItRMvlrh_6YQMA6-Y_ArefHMel80P1d_xQXFISLTTJM"
DOUBAO_API_KEY = "595dd44f-ae78-43ee-bbb6-2aaa23f2a17f"


def call_minimax_api(prompt, api_key, model="MiniMax-M2.7", max_retries=5, retry_delay=3):
    """
    调用MiniMax M2.7 API，包含重试机制
    """
    url = "https://api.minimaxi.com/v1/text/chatcompletion_v2"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000
    }

    for attempt in range(max_retries):
        try:
            start = time.time()
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            elapsed = time.time() - start

            # 检查是否过载
            if "overloaded" in resp.text or "529" in resp.text:
                print(f"    [MiniMax] 第{attempt+1}次尝试: 服务器过载，等待{retry_delay}秒后重试...")
                time.sleep(retry_delay)
                continue

            data = resp.json()
            if data.get("choices") and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                print(f"    [MiniMax] 成功 (耗时{elapsed:.1f}秒)")
                return content, elapsed, "success"
            else:
                print(f"    [MiniMax] 响应格式异常: {resp.text[:50]}")
                time.sleep(retry_delay)
                continue

        except requests.exceptions.Timeout:
            print(f"    [MiniMax] 第{attempt+1}次尝试: 超时，等待{retry_delay}秒后重试...")
            time.sleep(retry_delay)
            continue
        except Exception as e:
            print(f"    [MiniMax] 错误: {str(e)[:50]}")
            time.sleep(retry_delay)
            continue

    print(f"    [MiniMax] 最终失败: 达到最大重试次数")
    return None, 0, "failed"


def call_doubao_api(prompt, api_key, model="doubao-1-5-lite-32k-250115", max_retries=3, retry_delay=2):
    """
    调用Doubao API
    """
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000
    }

    for attempt in range(max_retries):
        try:
            start = time.time()
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            elapsed = time.time() - start

            data = resp.json()
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                print(f"    [Doubao] 成功 (耗时{elapsed:.1f}秒)")
                return content, elapsed, "success"
            else:
                print(f"    [Doubao] 响应异常: {resp.text[:50]}")
                return None, elapsed, "failed"

        except Exception as e:
            print(f"    [Doubao] 错误: {str(e)[:50]}")
            time.sleep(retry_delay)
            continue

    return None, 0, "failed"


# 定义测试用例
TEST_CASES = [
    {
        "id": 1,
        "name": "JSON格式输出",
        "prompt": """输入：肺活男:0.85;1000米男:1.10
任务：判断耐力等级
要求：直接返回JSON，不要有其他内容
格式：{"等级":"A/B/C","理由":"一句话"}""",
        "expected_format": "json",
        "expected_fields": ["等级", "理由"],
        "expected_length": "short",
        "constraint": "只返回JSON，无其他文字"
    },
    {
        "id": 2,
        "name": "严格数量约束",
        "prompt": """任务：推荐3门体育课程
输入：某学生体质测试数据
要求：必须exactly返回3门课程，不要多不要少
格式：课程1;课程2;课程3""",
        "expected_format": "semicolon_separated",
        "expected_fields": ["课程1", "课程2", "课程3"],
        "expected_length": "exact_3",
        "constraint": "恰好3门课程，用分号分隔"
    },
    {
        "id": 3,
        "name": "否定语义理解",
        "prompt": """输入：我不擅长球类运动，平时跑步比较多
任务：推荐适合的体育课程
要求：不要推荐任何球类课程（篮球/足球/排球/乒乓球/羽毛球/网球）
只返回课程名称，用逗号分隔""",
        "expected_format": "comma_separated",
        "expected_fields": ["课程列表"],
        "expected_length": "variable",
        "constraint": "不包含任何球类课程"
    },
    {
        "id": 4,
        "name": "仅返回指定内容",
        "prompt": """任务：根据体测数据判断体质类型
输入：BMI=22, 肺活量=4500, 1000米=240秒
要求：只返回体质类型（优秀/良好/及格/不及格），不要解释""",
        "expected_format": "single_word",
        "expected_fields": ["体质类型"],
        "expected_length": "single",
        "constraint": "只返回一个词"
    },
    {
        "id": 5,
        "name": "多字段JSON",
        "prompt": """输入：某学生体测数据
任务：生成完整的分析结果
要求：返回JSON，包含且仅包含以下4个字段：
- 体质等级: 字符串
- 优势项目: 字符串
- 劣势项目: 字符串
- 推荐课程: 字符串
不要有其他字段或解释""",
        "expected_format": "json",
        "expected_fields": ["体质等级", "优势项目", "劣势项目", "推荐课程"],
        "expected_length": "medium",
        "constraint": "JSON包含且仅包含4个字段"
    },
    {
        "id": 6,
        "name": "条件筛选",
        "prompt": """任务：从列表中筛选适合的课程
课程列表：篮球、足球、排球、田径、游泳、太极拳
条件：心肺耐力较差的学生
要求：只返回2门最合适的课程，用竖线分隔
格式：课程1|课程2""",
        "expected_format": "pipe_separated",
        "expected_fields": ["课程1", "课程2"],
        "expected_length": "exact_2",
        "constraint": "恰好2门课程，用|分隔"
    },
    {
        "id": 7,
        "name": "长度限制",
        "prompt": """任务：描述学生体质特点
输入：心肺耐力优秀，速度较快，但柔韧性较差
要求：描述控制在20个字以内（包含标点）
只返回描述，不要有前缀""",
        "expected_format": "limited_text",
        "expected_fields": ["描述"],
        "expected_length": "max_20",
        "constraint": "不超过20个字"
    },
    {
        "id": 8,
        "name": "多级筛选",
        "prompt": """任务：分层推荐课程
输入：男生，体测数据如下
步骤1：先判断体质类型
步骤2：根据体质类型推荐2门课程
要求：分两步返回，用"步骤1:"和"步骤2:"前缀
格式：
步骤1:体质类型
步骤2:课程1,课程2""",
        "expected_format": "two_step",
        "expected_fields": ["步骤1", "步骤2"],
        "expected_length": "two_lines",
        "constraint": "包含两个步骤，各有前缀"
    },
    {
        "id": 9,
        "name": "排除性约束",
        "prompt": """任务：为女生推荐课程
输入：力量较差，喜欢团队运动
要求：不要游泳
不要任何解释，只返回课程名""",
        "expected_format": "single_line",
        "expected_fields": ["课程名"],
        "expected_length": "short",
        "constraint": "不包含游泳，无解释"
    },
    {
        "id": 10,
        "name": "复合格式",
        "prompt": """任务：分析并推荐
输入：速度优秀，耐力较差
要求：按以下JSON格式返回，不要有其他内容
{
  "优势": "速度",
  "劣势": "耐力",
  "推荐理由": "一句话",
  "课程": "1门课程名"
}""",
        "expected_format": "json",
        "expected_fields": ["优势", "劣势", "推荐理由", "课程"],
        "expected_length": "medium",
        "constraint": "严格JSON格式，4个字段"
    },
    {
        "id": 11,
        "name": "序号约束",
        "prompt": """任务：排名3门最适合的课程
课程池：篮球、足球、田径、游泳、太极拳、羽毛球
输入：心肺耐力优秀，速度较快
要求：必须用序号格式返回：①课程名 ②课程名 ③课程名""",
        "expected_format": "numbered",
        "expected_fields": ["①", "②", "③"],
        "expected_length": "exact_3",
        "constraint": "使用①②③序号"
    },
    {
        "id": 12,
        "name": "顺序约束",
        "prompt": """任务：按推荐度排序3门课程
输入：体测数据
课程：篮球、田径、游泳
要求：按推荐度从高到低排序
格式：排名1的课|排名2的课|排名3的课""",
        "expected_format": "pipe_ranked",
        "expected_fields": ["排名1", "排名2", "排名3"],
        "expected_length": "exact_3",
        "constraint": "按顺序用|分隔"
    },
    {
        "id": 13,
        "name": "双重否定理解",
        "prompt": """任务：推荐课程
输入：不是耐力差的学生，不是喜欢球类的学生
要求：推荐适合耐力好且喜欢非球类运动的课程
只返回1门课程名，不要解释""",
        "expected_format": "single_word",
        "expected_fields": ["课程"],
        "expected_length": "single",
        "constraint": "只返回1门课程"
    },
    {
        "id": 14,
        "name": "空值处理",
        "prompt": """任务：如果数据不全，推荐保守型课程
输入：身高体重正常，但缺少心肺耐力数据
要求：如果无法判断就回复"无法推荐"，否则返回1门课程
只返回"无法推荐"或课程名""",
        "expected_format": "conditional",
        "expected_fields": ["结果"],
        "expected_length": "single",
        "constraint": "只返回固定格式"
    },
    {
        "id": 15,
        "name": "标签格式",
        "prompt": """任务：打标签
输入：某学生体测数据
要求：用以下标签格式返回，用分号分隔：
类型:类型名;难度:难度级;推荐:是或否
示例：类型:耐力型;难度:中等;推荐:是""",
        "expected_format": "tag_format",
        "expected_fields": ["类型", "难度", "推荐"],
        "expected_length": "short",
        "constraint": "严格标签格式"
    }
]


def score_format(response, expected_format):
    """评分：格式遵循"""
    if not response:
        return 0

    scores = {
        "json": 100 if response.strip().startswith('{') and response.strip().endswith('}') else 0,
        "semicolon_separated": 100 if ';' in response and '\n' not in response else 50,
        "comma_separated": 100 if ',' in response and ';' not in response and '{' not in response else 50,
        "single_word": 100 if len(response.strip()) < 10 and '\n' not in response and '{' not in response else 0,
        "pipe_separated": 100 if '|' in response else 0,
        "two_step": 100 if '步骤1' in response and '步骤2' in response else 0,
        "numbered": 100 if '①' in response or '1.' in response else 50,
        "pipe_ranked": 100 if '|' in response else 0,
        "conditional": 100 if response in ['无法推荐', '游泳', '田径'] or len(response.strip()) < 10 else 50,
        "tag_format": 100 if '类型:' in response and '难度:' in response and '推荐:' in response else 0,
        "limited_text": 50,
        "single_line": 50,
    }
    return scores.get(expected_format, 50)


def score_fields(response, expected_fields, expected_format):
    """评分：字段遵循"""
    if not response:
        return 0

    if expected_format == "json":
        try:
            data = json.loads(response)
            matched = sum(1 for f in expected_fields if f in data)
            return (matched / len(expected_fields)) * 100
        except:
            return 0
    else:
        matched = sum(1 for f in expected_fields if f in response)
        return (matched / len(expected_fields)) * 100


def score_content_constraint(response, constraint):
    """评分：内容约束（无多余文字）"""
    if not response:
        return 0

    unwanted = ["好的", "根据", "以下是", "下面是", "首先", "其次", "最后", "综上所述"]
    has_unwanted = any(u in response for u in unwanted)

    if '{' in response:
        try:
            json.loads(response)
            return 100
        except:
            return 50

    return 70 if not has_unwanted else 50


def score_length(response, expected_length):
    """评分：长度约束"""
    if not response:
        return 0

    word_count = len(response.strip())

    constraints = {
        "short": 100 if word_count < 20 else 50,
        "single": 100 if word_count < 10 else 50,
        "max_20": 100 if word_count <= 20 else 0,
        "exact_3": 100 if response.count('|') == 2 or response.count(';') == 2 or response.count(',') == 2 else 50,
        "exact_2": 100 if response.count('|') == 1 else 50,
        "medium": 100 if 10 < word_count < 100 else 50,
        "two_lines": 100 if '\n' in response and response.count('\n') == 1 else 50,
        "variable": 80,
    }
    return constraints.get(expected_length, 70)


def score_semantic(response, constraint, prompt):
    """评分：语义约束"""
    if not response:
        return 0

    constraint_lower = constraint.lower()

    if "不包含" in constraint or "不要" in constraint or "不要推荐" in constraint:
        forbidden = []
        if "篮球" in constraint or "球类" in constraint:
            forbidden = ["篮球", "足球", "排球", "乒乓球", "羽毛球", "网球"]
        if "游泳" in constraint:
            forbidden.append("游泳")

        for item in forbidden:
            if item in response:
                return 20

        return 100

    if "恰好" in constraint or "exactly" in constraint.lower():
        if "3门" in constraint or "3门" in prompt:
            return 100 if response.count(',') == 2 or response.count('|') == 2 or response.count(';') == 2 else 50
        if "2门" in constraint:
            return 100 if response.count(',') == 1 or response.count('|') == 1 else 50

    return 80


def test_api(model_name, api_func, prompt):
    """调用API并返回响应"""
    if model_name == "MiniMax-M2.7":
        return api_func(prompt, MINIMAX_API_KEY, model="MiniMax-M2.7", max_retries=5, retry_delay=5)
    else:
        return api_func(prompt, DOUBAO_API_KEY, model="doubao-1-5-lite-32k-250115", max_retries=3, retry_delay=2)


def run_compliance_test():
    """运行完整的遵循度测试"""
    results = []

    models = [
        ("MiniMax-M2.7", call_minimax_api),
        ("Doubao-lite", call_doubao_api)
    ]

    print("=" * 60)
    print("API遵循度对比测试")
    print("=" * 60)

    for model_name, api_func in models:
        print(f"\n【{model_name}测试中...】")

        for tc in TEST_CASES:
            print(f"  用例{tc['id']:2d}: {tc['name']}...", end=" ")
            response, elapsed, status = test_api(model_name, api_func, tc["prompt"])

            if response is None:
                response = ""
                status = "failed"

            format_score = score_format(response, tc["expected_format"]) if status == "success" else 0
            field_score = score_fields(response, tc["expected_fields"], tc["expected_format"]) if status == "success" else 0
            content_score = score_content_constraint(response, tc["constraint"]) if status == "success" else 0
            length_score = score_length(response, tc["expected_length"]) if status == "success" else 0
            semantic_score = score_semantic(response, tc["constraint"], tc["prompt"]) if status == "success" else 0

            overall = (format_score + field_score + content_score + length_score + semantic_score) / 5

            results.append({
                "模型": model_name,
                "用例ID": tc["id"],
                "用例名称": tc["name"],
                "格式遵循": format_score,
                "字段遵循": field_score,
                "内容约束": content_score,
                "长度约束": length_score,
                "语义理解": semantic_score,
                "综合得分": overall,
                "响应时间(秒)": elapsed,
                "状态": status,
                "响应内容": response[:80] if response else "无响应"
            })

            print(f"综合:{overall:5.1f}%")

            # 请求间隔，避免过快
            time.sleep(1)

    return pd.DataFrame(results)


def generate_report(df):
    """生成分析报告"""
    print("\n" + "=" * 60)
    print("遵循度测试报告")
    print("=" * 60)

    summary = df.groupby("模型").agg({
        "格式遵循": "mean",
        "字段遵循": "mean",
        "内容约束": "mean",
        "长度约束": "mean",
        "语义理解": "mean",
        "综合得分": "mean",
        "响应时间(秒)": "mean"
    }).round(2)

    print("\n【总体统计】")
    print(summary.to_string())

    dimensions = ["格式遵循", "字段遵循", "内容约束", "长度约束", "语义理解", "综合得分"]
    print("\n【各维度对比】")
    for dim in dimensions:
        scores = {}
        for model in df["模型"].unique():
            scores[model] = df[df["模型"] == model][dim].mean()

        if len(scores) == 2:
            model_names = list(scores.keys())
            winner = f"{model_names[0]}领先" if scores[model_names[0]] > scores[model_names[1]] else \
                     f"{model_names[1]}领先" if scores[model_names[1]] > scores[model_names[0]] else "平局"
            print(f"  {dim}: {scores[model_names[0]]:.1f}% vs {scores[model_names[1]]:.1f}% → {winner}")
        else:
            for model, score in scores.items():
                print(f"  {dim}: {model}={score:.1f}%")

    return summary


def plot_comparison(df, summary):
    """绘制对比图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    dimensions = ["格式遵循", "字段遵循", "内容约束", "长度约束", "语义理解"]

    models = df["模型"].unique().tolist()
    colors = ['#2ecc71', '#3498db'][:len(models)]

    # 图1: 雷达图
    ax1 = fig.add_subplot(221, polar=True)
    angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]

    for i, model in enumerate(models):
        scores = [df[df["模型"] == model][d].mean() for d in dimensions]
        scores += scores[:1]
        ax1.plot(angles, scores, 'o-', linewidth=2, label=model, color=colors[i])
        ax1.fill(angles, scores, alpha=0.25, color=colors[i])

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(dimensions, fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.set_title('各维度遵循度对比', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 图2: 柱状图 - 总体得分
    ax2 = axes[0, 1]
    overall_scores = [summary.loc[model, "综合得分"] for model in models]
    bars = ax2.bar(models, overall_scores, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('遵循度 (%)', fontsize=11)
    ax2.set_title('总体遵循度对比', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    for bar, score in zip(bars, overall_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{score:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 图3: 分项得分对比
    ax3 = axes[1, 0]
    x = np.arange(len(dimensions))
    width = 0.35

    for i, model in enumerate(models):
        scores = [df[df["模型"] == model][d].mean() for d in dimensions]
        offset = (i - len(models)/2 + 0.5) * width
        ax3.bar(x + offset, scores, width, label=model, color=colors[i])

    ax3.set_ylabel('遵循度 (%)', fontsize=11)
    ax3.set_title('各维度分项得分', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(dimensions, fontsize=9, rotation=15)
    ax3.legend()
    ax3.set_ylim(0, 100)

    # 图4: 响应时间对比
    ax4 = axes[1, 1]
    times = [summary.loc[model, "响应时间(秒)"] for model in models]
    bars = ax4.bar(models, times, color=colors, edgecolor='black', linewidth=1.2)
    ax4.set_ylabel('响应时间 (秒)', fontsize=11)
    ax4.set_title('API响应时间对比', fontsize=14, fontweight='bold')
    for bar, t in zip(bars, times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{t:.2f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/Users/bytedance/Desktop/毕设/data/compliance_comparison.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存: /Users/bytedance/Desktop/毕设/data/compliance_comparison.png")

    df.to_csv('/Users/bytedance/Desktop/毕设/data/compliance_test_results.csv', index=False, encoding='utf-8-sig')
    summary.to_csv('/Users/bytedance/Desktop/毕设/data/compliance_summary.csv', encoding='utf-8-sig')
    print("数据已保存: compliance_test_results.csv, compliance_summary.csv")


def main():
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['STHei Light.ttc', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    df = run_compliance_test()
    summary = generate_report(df)
    plot_comparison(df, summary)

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()