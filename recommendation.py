"""
AI推荐模块
所有类型判断、优劣势分析、课程推荐均由MiniMax LLM API完成
"""

import requests
import json
import time
from courses import SPORT_COURSES


def call_minimax_api_stream(prompt, api_key, model="MiniMax-M2.7"):
    """
    调用MiniMax API（流式输出）

    Args:
        prompt: 输入提示词
        api_key: MiniMax API密钥
        model: 使用的模型

    Yields:
        逐段返回AI生成的内容
    """
    if not api_key:
        yield "error: API Key未配置"
        return

    try:
        url = "https://api.minimaxi.com/anthropic/v1/messages"

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "stream": True
        }

        response = requests.post(url, headers=headers, json=payload, timeout=120, stream=True)
        response.raise_for_status()

        # 处理SSE流
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data:'):
                    data = line[5:].strip()
                    if data and data != '[DONE]':
                        try:
                            json_data = json.loads(data)
                            # 找到text类型的content
                            if "content" in json_data:
                                for item in json_data["content"]:
                                    if item.get("type") == "text":
                                        yield item["text"]
                        except json.JSONDecodeError:
                            continue

    except requests.exceptions.Timeout:
        yield "error: 请求超时"
    except requests.exceptions.RequestException as e:
        yield f"error: 请求失败: {str(e)}"
    except Exception as e:
        yield f"error: 调用出错: {str(e)}"


def call_minimax_api(prompt, api_key, model="doubao-1-5-lite-32k-250115"):
    """
    调用火山引擎 Doubao API（OpenAI兼容格式）

    Args:
        prompt: 输入提示词
        api_key: 火山引擎 API密钥
        model: 使用的模型

    Returns:
        AI生成的回复内容，失败返回None
    """
    import time

    if not api_key:
        print("错误: API Key为空")
        return None

    api_start = time.time()
    try:
        # 火山引擎 OpenAI 兼容接口
        url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000
        }

        response = requests.post(url, headers=headers, json=payload, timeout=120)
        api_time = time.time() - api_start
        print(f"[火山引擎 API] 耗时: {api_time:.1}秒, 状态: {response.status_code}")

        response.raise_for_status()

        result = response.json()

        # OpenAI 格式：choices[0].message.content
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0].get("message", {}).get("content", "")
            if content:
                print(f"[火山引擎 API] 找到内容，长度: {len(content)}")
                return content

        print(f"[火山引擎 API] 未找到内容. result_keys: {list(result.keys())}")
        return None

    except requests.exceptions.Timeout:
        print(f"[火山引擎 API] 请求超时 (120秒)")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[火山引擎 API] 请求失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应内容: {e.response.text[:500]}")
        return None
    except Exception as e:
        print(f"[火山引擎 API] 调用出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_cluster_with_llm(cluster_data, api_key):
    """
    使用LLM分析聚类结果，生成群体类型、描述、优劣势

    Args:
        cluster_data: 聚类数据，包含指标均值等信息
        api_key: API密钥

    Returns:
        分析结果字典，包含cluster_type, description, weak_features, strong_features
    """
    if not api_key:
        print("错误: analyze_cluster_with_llm 收到空API Key")
        return None

    # 构建提示词
    indicator_info = []
    for indicator, stats in cluster_data.items():
        mean_val = stats.get("mean", 0)
        std_val = stats.get("std", 0)
        gender = stats.get("gender", "")
        indicator_info.append(f"- {indicator}({gender}): 均值={mean_val:.1f}, 标准差={std_val:.1f}")

    prompt = f"""你是一个体育教育数据分析专家。请分析以下学生群体的体测数据，判断其类型特征。

群体数据：
{chr(10).join(indicator_info)}

请以JSON格式返回分析结果，包含以下字段：
- cluster_type: 群体类型名称（如"力量型"、"耐力薄弱型"、"均衡型"等）
- description: 一段话描述该群体的特点，包含性别比例、整体身体素质评价
- weak_features: 薄弱项列表（数组），如["心肺耐力", "柔韧性"]
- strong_features: 优势项列表（数组），如["力量", "爆发力"]

注意：
1. 判断基于指标数据，与全国平均值对比
2. 1000米跑/800米跑成绩差表示耐力差（数值越大越差）
3. 肺活量低表示心肺功能弱
4. 引体向上/仰卧起坐差表示力量差
5. 立定跳远近表示爆发力差
6. 坐位体前屈数值小表示柔韧性差

请直接返回JSON，不要有其他内容。"""

    print(f"开始调用LLM分析群体类型，指标数量: {len(cluster_data)}")
    response = call_minimax_api(prompt, api_key)
    if not response:
        print("错误: LLM返回空响应")
        return None

    try:
        # 尝试解析JSON
        # 清理响应文本，提取JSON部分
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        result = json.loads(response)
        print(f"LLM分析成功: cluster_type={result.get('cluster_type')}")
        return result
    except json.JSONDecodeError as e:
        print(f"LLM返回非JSON格式: {e}, 原始响应: {response[:500]}")
        return None


def recommend_courses_with_llm(cluster_analysis, api_key):
    """
    使用LLM根据群体分析结果推荐课程

    Args:
        cluster_analysis: analyze_cluster_with_llm返回的分析结果
        api_key: API密钥

    Returns:
        推荐结果列表
    """
    if not api_key:
        print("错误: recommend_courses_with_llm 收到空API Key")
        return None

    # 构建课程信息
    course_info = [f"- {c['name']}: {c['target']} (属于{c['category']})" for c in SPORT_COURSES]

    prompt = f"""你是体育课程推荐专家。基于以下学生群体分析结果，推荐最合适的体育课程。

群体分析：
类型：{cluster_analysis.get('cluster_type', '未知')}
描述：{cluster_analysis.get('description', '')}
薄弱项：{', '.join(cluster_analysis.get('weak_features', []))}
优势项：{', '.join(cluster_analysis.get('strong_features', []))}

可选课程：
{chr(10).join(course_info)}

请以JSON格式返回推荐结果，包含top3推荐课程，每门课程包含：
- course_name: 课程名称
- course_category: 课程类别
- score: 匹配度评分（0-100）
- reason: 推荐理由，说明该课程如何适配该群体

请直接返回JSON数组，不要有其他内容。"""

    print(f"开始调用LLM生成课程推荐 for {cluster_analysis.get('cluster_type', '未知')}")
    response = call_minimax_api(prompt, api_key)
    if not response:
        print("错误: 课程推荐LLM返回空响应")
        return None

    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        result = json.loads(response)
        print(f"课程推荐成功，返回{len(result) if isinstance(result, list) else 0}条")
        return result if isinstance(result, list) else []
    except json.JSONDecodeError as e:
        print(f"课程推荐LLM返回非JSON格式: {e}")
        return None


def analyze_and_recommend_cluster(cluster_data, api_key):
    """
    使用LLM同时分析聚类结果并生成课程推荐（极致优化版）
    """
    if not api_key:
        return None

    # 精简指标数据
    indicator_info = []
    for indicator, stats in cluster_data.items():
        mean_val = stats.get("mean", 0)
        gender = stats.get("gender", "")
        indicator_info.append(f"{indicator}({gender[0]}):{mean_val:.0f}")

    # 精简课程信息 - 只保留名称和目标
    course_info = [f"{c['name']}:{c['target']}" for c in SPORT_COURSES]

    prompt = f"""分析学生群体并推荐课程。

数据:{';'.join(indicator_info)}

课程:{';'.join(course_info)}

返回JSON格式:
{{"t":"类型","d":"描述","w":["弱项"],"s":["强项"],"r":[{{"n":"课程名","s":分数,"y":"原因"}}]}}

直接返回JSON。"""

    response = call_minimax_api(prompt, api_key)
    if not response:
        return None

    try:
        # 清理响应
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        result = json.loads(response)
        # 映射简短键名到完整键名
        return {
            "cluster_type": result.get("t", "未知"),
            "description": result.get("d", ""),
            "weak_features": result.get("w", []),
            "strong_features": result.get("s", []),
            "recommendations": [
                {
                    "course_name": r.get("n", ""),
                    "score": r.get("s", 0),
                    "reason": r.get("y", "")
                }
                for r in result.get("r", [])
            ]
        }
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        return None


def batch_analyze_types(batch_data):
    """
    Step 1: 批量判断所有群体类型
    输入: {cluster_id: {"relative_data": {}, "label": "C1"}}
    输出: "C1:力量型,C2:耐力型" 格式的字符串
    """
    parts = []
    for cid, data in batch_data.items():
        label = data.get("label", f"C{cid+1}")
        rel = data.get("relative_data", {})

        # 构建筑居数据
        rel_parts = []
        for key, val in rel.items():
            indicator = key.rsplit("_", 1)[0][:2]
            gender = key.rsplit("_", 1)[1][0] if "_" in key else "?"
            rel_val = val.get("relative", 1.0)
            rel_parts.append(f"{indicator}{gender}:{rel_val:.2f}")

        data_str = ";".join(rel_parts)
        parts.append(f"{label}({data_str})")

    prompt = f"判断以下群体类型，只返回格式：群体:类型,群体:类型\n数据:{','.join(parts)}\n规则:1000m大=耐力差,肺活量小=心肺弱,引体向上少=力量差,50m大=速度差,立定跳远远=爆发力好,坐位体前屈小=柔韧性差"
    return prompt


def batch_analyze_strengths(batch_data):
    """
    Step 2: 批量判断所有群体优势
    输入: 同上
    输出: "C1:优势1|优势2,C2:优势1" 格式
    """
    parts = []
    for cid, data in batch_data.items():
        label = data.get("label", f"C{cid+1}")
        rel = data.get("relative_data", {})

        rel_parts = []
        for key, val in rel.items():
            indicator = key.rsplit("_", 1)[0][:2]
            gender = key.rsplit("_", 1)[1][0] if "_" in key else "?"
            rel_val = val.get("relative", 1.0)
            rel_parts.append(f"{indicator}{gender}:{rel_val:.2f}")

        data_str = ";".join(rel_parts)
        parts.append(f"{label}({data_str})")

    prompt = f"判断以下群体优势项，只返回格式：群体:优势1|优势2,群体:优势1\n数据:{','.join(parts)}\n规则:1000m大=耐力差,肺活量小=心肺弱,引体向上少=力量差,50m大=速度差,立定跳远远=爆发力好,坐位体前屈小=柔韧性差"
    return prompt


def batch_analyze_weaknesses(batch_data):
    """
    Step 3: 批量判断所有群体劣势
    输入: 同上
    输出: "C1:劣势1|劣势2,C2:劣势1" 格式
    """
    parts = []
    for cid, data in batch_data.items():
        label = data.get("label", f"C{cid+1}")
        rel = data.get("relative_data", {})

        rel_parts = []
        for key, val in rel.items():
            indicator = key.rsplit("_", 1)[0][:2]
            gender = key.rsplit("_", 1)[1][0] if "_" in key else "?"
            rel_val = val.get("relative", 1.0)
            rel_parts.append(f"{indicator}{gender}:{rel_val:.2f}")

        data_str = ";".join(rel_parts)
        parts.append(f"{label}({data_str})")

    prompt = f"判断以下群体劣势项，只返回格式：群体:劣势1|劣势2,群体:劣势1\n数据:{','.join(parts)}\n规则:1000m大=耐力差,肺活量小=心肺弱,引体向上少=力量差,50m大=速度差,立定跳远远=爆发力好,坐位体前屈小=柔韧性差"
    return prompt


def parse_batch_response(text, key_name):
    """
    解析批量响应结果
    text: "C1:类型1,C2:类型2" 或 "C1:优势1|优势2,C2:优势1"
    key_name: "type" | "strong" | "weak"
    返回: {cluster_id: value}
    """
    import re
    results = {}

    # 匹配 C1:xxx 或 C1(男):xxx 格式
    pattern = r'([A-Za-z0-9_\-]+)\s*:\s*([^,]+)'
    matches = re.findall(pattern, text)

    for label, value in matches:
        # 提取群体编号
        cid_match = re.search(r'(\d+)', label)
        if cid_match:
            cid = int(cid_match.group(1)) - 1  # 转为0索引

        if key_name == "type":
            results[cid] = value.strip()
        else:
            # 分割优势/劣势列表
            items = [s.strip() for s in value.split('|') if s.strip()]
            results[cid] = items

    return results


def parse_llm_response(text):
    """
    从LLM输出的文本中解析群体分析结果
    使用规则匹配而非JSON解析

    Args:
        text: LLM输出的原始文本

    Returns:
        {cluster_id: {"cluster_type", "description", "weak_features", "strong_features"}}
    """
    import re

    results = {}

    # 按[群体X]分段
    # 匹配 [群体1] 或 [群体2] 等
    pattern = r'\[群体(\d+)\]\s*\n(.*?)(?=\[群体\d+\]|$)'
    matches = re.findall(pattern, text, re.DOTALL)

    for cid_str, content in matches:
        cid = int(cid_str)

        # 提取类型
        type_match = re.search(r'类型:\s*(.+)', content)
        cluster_type = type_match.group(1).strip() if type_match else "待评估"

        # 提取描述
        desc_match = re.search(r'描述:\s*(.+)', content)
        description = desc_match.group(1).strip() if desc_match else ""

        # 提取优势列表
        strong_match = re.search(r'优势:\s*(.+)', content)
        strong_features = []
        if strong_match:
            strong_str = strong_match.group(1).strip()
            # 按逗号分割，去除空白
            strong_features = [s.strip() for s in strong_str.split(',') if s.strip()]

        # 提取劣势列表
        weak_match = re.search(r'劣势:\s*(.+)', content)
        weak_features = []
        if weak_match:
            weak_str = weak_match.group(1).strip()
            weak_features = [s.strip() for s in weak_str.split(',') if s.strip()]

        results[cid] = {
            "cluster_type": cluster_type,
            "description": description,
            "strong_features": strong_features,
            "weak_features": weak_features
        }

    print(f"[解析] 成功解析 {len(results)} 个群体")
    return results


def analyze_all_clusters_with_llm(batch_data, baselines, api_key):
    """
    一次调用分析所有聚类，传入相对水平数据

    Args:
        batch_data: {cluster_id: {"label", "dominant_gender", "relative_data"}}
        baselines: {indicator_gender: baseline_value}
        api_key: API密钥

    Returns:
        {cluster_id: {"cluster_type", "description", "weak_features", "strong_features"}}
    """
    if not api_key:
        return None

    # 构建紧凑的输入数据
    cluster_lines = []
    for cid, data in batch_data.items():
        label = data["label"]
        gender = data["dominant_gender"][0] if data["dominant_gender"] else "?"
        relative = data["relative_data"]

        # 提取相对水平，格式：指标简称_性别:相对值
        rel_parts = []
        for key, val in relative.items():
            # key格式: "肺活量_男生"
            parts = key.rsplit("_", 1)
            indicator = parts[0][:2]  # 取前2字简称
            g = parts[1][0] if len(parts) > 1 else "?"
            rel = val.get("relative", 1.0)
            rel_parts.append(f"{indicator}{g}:{rel:.2f}")

        rel_str = ";".join(rel_parts)
        cluster_lines.append(f"{label}({gender}):{rel_str}")

    data_str = ';'.join(cluster_lines)
    prompt = "分析群体:" + data_str + "\n规则:1000m大=耐力差,肺活量小=心肺弱,引体向上少=力量差,仰卧起坐少=力量差,立定跳远远=爆发力好,50m大=速度差,坐位体前屈小=柔韧性差\n\n输出格式(直接输出，不要思考)：\n[群体1]\n类型: XXX\n描述: XXX\n优势: 项1, 项2\n劣势: 项3, 项4\n\n[群体2]\n类型: XXX\n描述: XXX\n优势: 项1, 项2\n劣势: 项3, 项4"

    print(f"[LLM批量分析] 开始调用API，输入长度: {len(prompt)}")
    api_start = time.time()

    response = call_minimax_api(prompt, api_key)

    api_time = time.time() - api_start
    print(f"[LLM批量分析] API耗时: {api_time:.1f}秒")

    if not response:
        print(f"[LLM批量分析] API返回空")
        return None

    try:
        response = response.strip()
        print(f"[LLM批量分析] 原始响应:\n{response}\n---")

        # 使用规则解析而非JSON
        result = parse_llm_response(response)
        print(f"[LLM批量分析] 解析成功，返回 {len(result)} 个聚类")
        return result

    except Exception as e:
        print(f"[LLM批量分析] 解析失败: {e}")
        print(f"[LLM批量分析] 原始响应: {response[:500]}")
        return None


def analyze_single_cluster(relative_data, dominant_gender, api_key):
    """
    分析单个聚类，返回类型、描述、优劣势
    使用文本格式+规则解析，避免JSON截断问题

    Args:
        relative_data: {indicator_gender: {"relative": float}}
        dominant_gender: 主导性别
        api_key: API密钥

    Returns:
        {"cluster_type", "description", "weak_features", "strong_features"}
    """
    if not api_key:
        return None

    # 构建紧凑数据
    parts = []
    for key, val in relative_data.items():
        indicator = key.rsplit("_", 1)[0][:2]
        gender = key.rsplit("_", 1)[1][0] if "_" in key else "?"
        rel = val.get("relative", 1.0)
        parts.append(f"{indicator}{gender}:{rel:.2f}")

    data_str = ";".join(parts)
    gender_str = dominant_gender[0] if dominant_gender else "?"

    prompt = f"群体({gender_str}):{data_str}\n规则:1000m大=耐力差,肺活量小=心肺弱,引体向上少=力量差,仰卧起坐少=力量差,立定跳远远=爆发力好,50m大=速度差,坐位体前屈小=柔韧性差\n输出(直接):类型:XXX|描述:XXX|优势:项1,项2|劣势:项3"

    response = call_minimax_api(prompt, api_key)
    if not response:
        return None

    # 规则解析
    result = {
        "cluster_type": "待评估",
        "description": "",
        "weak_features": [],
        "strong_features": []
    }

    try:
        text = response.strip()
        print(f"[单群体分析] 响应: {text[:200]}")

        # 解析类型
        type_m = re.search(r'类型:\s*(.+?)(?:\|描述:|$)', text)
        if type_m:
            result["cluster_type"] = type_m.group(1).strip()

        # 解析描述
        desc_m = re.search(r'描述:\s*(.+?)(?:\|优势:|$)', text)
        if desc_m:
            result["description"] = desc_m.group(1).strip()

        # 解析优势
        strong_m = re.search(r'优势:\s*(.+?)(?:\|劣势:|$)', text)
        if strong_m:
            strong_str = strong_m.group(1).strip()
            result["strong_features"] = [s.strip() for s in strong_str.split(',') if s.strip()]

        # 解析劣势
        weak_m = re.search(r'劣势:\s*(.+?)(?:\|描述:|$)', text)
        if weak_m:
            weak_str = weak_m.group(1).strip()
            result["weak_features"] = [s.strip() for s in weak_str.split(',') if s.strip()]

        print(f"[单群体分析] 解析结果: {result['cluster_type']}")
        return result

    except Exception as e:
        print(f"[单群体分析] 解析失败: {e}, 响应: {response[:200]}")
        return None


def analyze_all_clusters_with_llm(batch_data, baselines, api_key):
        print(f"[LLM批量分析] 解析失败: {e}")
        print(f"[LLM批量分析] 原始响应: {response[:500]}")
        return None


def generate_course_decision_with_llm(cluster_analyses, course_distribution, api_key):
    """
    使用LLM生成课程配置决策建议

    Args:
        cluster_analyses: 各聚类的分析结果列表
        course_distribution: 现有课程分布
        api_key: API密钥

    Returns:
        决策建议（文本）
    """
    if not api_key:
        return None

    # 构建群体分析摘要
    cluster_summary = []
    for i, analysis in enumerate(cluster_analyses):
        weak = ', '.join(analysis.get('weak_features', []))
        strong = ', '.join(analysis.get('strong_features', []))
        ratio = analysis.get('ratio', 0)
        cluster_summary.append(f"- 群体{i+1}(占比{ratio}%): 类型={analysis.get('cluster_type', '未知')}, 薄弱项={weak}, 优势项={strong}")

    # 构建课程分布
    dist_summary = [f"- {cat}: {count}门" for cat, count in course_distribution.items()]

    prompt = f"""你是学校体育课程规划专家。基于学生群体的体测分析结果，给出课程配置建议。

学生群体分析：
{chr(10).join(cluster_summary)}

现有课程分布：
{chr(10).join(dist_summary)}

请给出课程类型调整建议：
1. 分析各群体最需要的课程类型
2. 对比现有课程分布，指出供需矛盾
3. 建议增加/维持/缩减哪些类型的课程
4. 说明理由

请用清晰的中文回答，建议控制在200字以内。"""

    return call_minimax_api(prompt, api_key)


def allocate_courses_smartly(cluster_features, course_recommendations, total_students):
    """
    智能课程分配算法 - 解决课程扎堆问题

    策略：
    1. 大群体拆散到多门课（避免单门课压力过大）
    2. 确保每门课都有学生上
    3. 优先满足高匹配度的分配

    Args:
        cluster_features: 聚类特征，包含每个群体的大小
        course_recommendations: {cluster_id: [{"course_name": str, "score": float}]}
        total_students: 学生总数

    Returns:
        {cluster_id: [allocated_courses]}
    """
    from courses import SPORT_COURSES

    n_courses = len(SPORT_COURSES)
    n_clusters = len(cluster_features)

    # 每门课的容量限制（最多接受总学生的20%，最少5%）
    MAX_RATIO = 0.25  # 单门课最多25%
    MIN_RATIO = 0.03  # 单门课最少3%（确保每门课都有学生）

    max_capacity = int(total_students * MAX_RATIO)
    min_capacity = int(total_students * MIN_RATIO)

    # 当前每门课的已分配人数
    course_allocated = {c["name"]: 0 for c in SPORT_COURSES}

    # 每门课的目标分配（用于多样性）
    # 初始目标：平均分配，但可以根据实际情况调整
    course_targets = {c["name"]: 0 for c in SPORT_COURSES}

    result = {}

    # 按群体大小降序排序（先处理大群体）
    sorted_clusters = sorted(
        cluster_features.items(),
        key=lambda x: x[1].get("size", 0),
        reverse=True
    )

    for cluster_id, cluster_info in sorted_clusters:
        cluster_size = cluster_info.get("size", 0)
        recs = course_recommendations.get(cluster_id, [])

        if not recs or cluster_size == 0:
            result[cluster_id] = []
            continue

        # 该群体需要分配的课程数（根据群体大小动态调整）
        # 小群体(1-5%)分配1-2门课，中等群体(5-15%)分配2-3门课，大群体(15%+)分配3-4门课
        cluster_ratio = cluster_size / total_students
        if cluster_ratio >= 0.15:
            num_courses = min(4, len(recs))
        elif cluster_ratio >= 0.05:
            num_courses = min(3, len(recs))
        else:
            num_courses = min(2, len(recs))

        allocated = []
        remaining_size = cluster_size

        for i, rec in enumerate(recs[:num_courses]):
            course_name = rec.get("course_name")
            base_score = rec.get("score", 70)

            if not course_name or remaining_size <= 0:
                break

            current_allocated = course_allocated.get(course_name, 0)

            # 检查该课程是否还有余量
            # 如果课程已满，尝试分配其他课程
            if current_allocated >= max_capacity:
                # 找下一门匹配度较高的课程
                continue

            # 计算实际可分配的人数
            # 考虑：当前容量、多样性（避免完全平均）、匹配度
            available = max_capacity - current_allocated

            if available <= 0:
                continue

            # 对于大群体中的学生，分散到多门课
            # 每门课分配的人数不超过该群体剩余人数的60%
            max_to_allocate = min(
                available,
                int(remaining_size * 0.6) if i > 0 else available,
                max(int(total_students * 0.05), 10)  # 至少10人或5%
            )

            if max_to_allocate > 0:
                allocated.append({
                    "course_name": course_name,
                    "allocated_count": max_to_allocate,
                    "score": base_score,
                    "reason": rec.get("reason", "")
                })
                course_allocated[course_name] += max_to_allocate
                remaining_size -= max_to_allocate

        # 如果还有剩余学生没分配（因为课程满了），强制分配到有空间的课程
        if remaining_size > 0:
            for rec in recs:
                if remaining_size <= 0:
                    break
                course_name = rec.get("course_name")
                if course_allocated.get(course_name, 0) < max_capacity:
                    alloc = min(remaining_size, int(total_students * 0.03))
                    allocated.append({
                        "course_name": course_name,
                        "allocated_count": alloc,
                        "score": rec.get("score", 70),
                        "reason": rec.get("reason", "")
                    })
                    course_allocated[course_name] += alloc
                    remaining_size -= alloc

        result[cluster_id] = allocated

    # 确保每门课都有学生（如果有课程没被分配到任何学生）
    unallocated_courses = [c["name"] for c in SPORT_COURSES if course_allocated.get(c["name"], 0) == 0]

    if unallocated_courses and result:
        # 找到最小的群体，把没人上的课程分配给他们
        smallest_cluster = min(result.items(), key=lambda x: sum(a["allocated_count"] for a in x[1]))
        if smallest_cluster[1]:
            # 从该群体中拿出一部分学生分配给没人上的课
            for course_name in unallocated_courses:
                if smallest_cluster[1]:
                    # 从第一个分配中分10%出去
                    first_alloc = smallest_cluster[1][0]
                    if first_alloc["allocated_count"] > 20:
                        taken = int(first_alloc["allocated_count"] * 0.1)
                        first_alloc["allocated_count"] -= taken
                        course_allocated[course_name] += taken
                        smallest_cluster[1].append({
                            "course_name": course_name,
                            "allocated_count": taken,
                            "score": 50,
                            "reason": "平衡课程分布"
                        })

    return result


def get_allocation_summary(cluster_features, allocation_result, total_students):
    """
    获取分配结果摘要
    """
    summary = {
        "total_students": total_students,
        "course_distribution": {},
        "cluster_distribution": {}
    }

    # 统计每门课的学生数
    for cluster_id, allocations in allocation_result.items():
        cluster_size = cluster_features.get(cluster_id, {}).get("size", 0)
        summary["cluster_distribution"][cluster_id] = {
            "total": cluster_size,
            "allocated": sum(a["allocated_count"] for a in allocations)
        }

        for alloc in allocations:
            course_name = alloc["course_name"]
            if course_name not in summary["course_distribution"]:
                summary["course_distribution"][course_name] = 0
            summary["course_distribution"][course_name] += alloc["allocated_count"]

    return summary


if __name__ == "__main__":
    # 测试用
    test_data = {
        "肺活量": {"mean": 3200, "std": 500, "gender": "男"},
        "1000米跑": {"mean": 280, "std": 30, "gender": "男"},
        "引体向上": {"mean": 5, "std": 3, "gender": "男"},
        "立定跳远": {"mean": 210, "std": 20, "gender": "男"},
    }

    print("测试LLM分析功能...")
    # result = analyze_cluster_with_llm(test_data, "your_api_key")
    # print(result)
