"""
体测数据驱动的大语言模型体育课程个性化推荐系统
Flask主服务器
"""

import os
import json
import threading
import time
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np

from preprocessing import preprocess_pipeline, validate_columns
from clustering import clustering_pipeline
from recommendation import (
    recommend_courses_with_llm,
    generate_course_decision_with_llm,
    call_minimax_api_stream,
    call_minimax_api
)
from flask import Response
from courses import SPORT_COURSES
from config import MINIMAX_API_KEY

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)


def convert_to_native_types(obj):
    """将numpy类型转换为Python原生类型"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# 全局状态存储
system_state = {
    "preprocessing_result": None,
    "clustering_result": None,
    "current_labels": None,
    "current_students": None,
    "analysis_done": False,      # AI分析是否已完成
    "analysis_result": None,    # AI分析结果
    "chat_history": []          # 聊天历史
}


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/courses', methods=['GET'])
def get_courses():
    """获取所有课程列表"""
    return jsonify({
        "code": 0,
        "data": SPORT_COURSES
    })


@app.route('/api/reset', methods=['POST'])
def reset_state():
    """刷新页面时清除后端状态"""
    system_state["preprocessing_result"] = None
    system_state["clustering_result"] = None
    system_state["current_labels"] = None
    system_state["current_students"] = None
    system_state["analysis_done"] = False
    system_state["analysis_result"] = None
    system_state["chat_history"] = []
    return jsonify({"code": 0, "message": "状态已重置"})


@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    """
    数据预处理接口
    """
    if 'file' not in request.files:
        return jsonify({"code": 1, "message": "请上传文件"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"code": 1, "message": "文件名为空"})

    try:
        # 保留原始文件扩展名
        ext = os.path.splitext(file.filename)[1] or '.xlsx'
        upload_path = os.path.join('data', f'uploaded_file{ext}')
        os.makedirs('data', exist_ok=True)
        file.save(upload_path)

        result = preprocess_pipeline(upload_path)
        # 上传新文件时清除所有旧状态
        system_state["preprocessing_result"] = result
        system_state["clustering_result"] = None
        system_state["current_labels"] = None
        system_state["current_students"] = None
        system_state["analysis_done"] = False
        system_state["analysis_result"] = None
        system_state["chat_history"] = []

        df = result["filtered_data"]
        validation = result["validation"]

        preview_df = df.head(5).fillna("")
        # 转换所有值为原生Python类型，处理numpy类型
        sample_preview = []
        for _, row in preview_df.iterrows():
            sample_preview.append({k: (None if pd.isna(v) else v) for k, v in row.items()})

        overview = {
            "total_samples": len(df),
            "valid_columns": list(validation["indicator_columns"].keys()),
            "gender_column": validation["gender_column"],
            "sample_preview": sample_preview
        }

        return jsonify({
            "code": 0,
            "message": "预处理成功",
            "data": overview
        })

    except Exception as e:
        return jsonify({"code": 1, "message": f"预处理失败: {str(e)}"})


@app.route('/api/cluster', methods=['POST'])
def cluster():
    """
    聚类分析接口（仅做聚类，不包含LLM分析）
    """
    if system_state["preprocessing_result"] is None:
        return jsonify({"code": 1, "message": "请先上传并预处理数据"})

    def generate():
        import traceback
        try:
            result = system_state["preprocessing_result"]
            df = result["filtered_data"]
            validation = result["validation"]
            indicator_cols = validation["indicator_columns"]

            print(f"[聚类] 数据形状: {df.shape}, 指标列: {list(indicator_cols.keys())}")

            # 步骤1: 准备数据
            yield 'data: {"step": "preparing", "message": "正在准备聚类数据..."}\n\n'
            from clustering import prepare_clustering_data, find_optimal_clusters, extract_cluster_features
            X, norm_columns = prepare_clustering_data(df, validation, indicator_cols)
            print(f"[聚类] 聚类特征矩阵形状: {X.shape}, 列: {norm_columns}")

            # 步骤2: HDBSCAN/GMM自动确定最优聚类数并聚类（使用分层聚类）
            yield 'data: {"step": "finding_k", "message": "正在使用分层密度聚类分析..."}\n\n'
            labels, elbow_plot, n_clusters = find_optimal_clusters(
                X, min_clusters=3, max_clusters=8, df=df, gender_col=validation["gender_column"]
            )
            print(f"[聚类] 自动确定聚类数: {n_clusters}")

            # 步骤3: 聚类已完成（上面一步完成）
            yield 'data: {"step": "clustering", "message": "聚类完成，正在提取群体特征..."}\n\n'
            print(f"[聚类] 聚类完成，标签数量: {len(labels)}")

            # 步骤4: 提取群体特征（使用标准化数据）
            yield 'data: {"step": "extracting", "message": "正在提取群体特征..."}\n\n'
            df_norm = result["normalized_data"]
            cluster_features, baselines = extract_cluster_features(
                df_norm, labels, validation, indicator_cols
            )
            print(f"[聚类] 提取到 {len(cluster_features)} 个群体特征")
            print(f"[聚类] baselines: {baselines}")

            # 保存聚类结果（不含LLM分析）
            system_state["clustering_result"] = {
                "optimal_k": int(n_clusters),
                "labels": [int(l) for l in labels],
                "cluster_features": cluster_features,
                "baselines": baselines,
                "elbow_plot": elbow_plot,
                "analyzed": False
            }
            system_state["current_labels"] = [int(l) for l in labels]
            system_state["current_students"] = df.to_dict(orient='records')

            # 构建返回数据（简化版，不包含indicator_data和base64图片）
            cluster_overview = []
            for cid, info in cluster_features.items():
                cluster_overview.append({
                    "cluster_id": int(cid),
                    "label": str(info.get("label", f"C{cid+1}")),
                    "ratio": float(info.get("ratio", 0)),
                    "size": int(info.get("size", 0)),
                    "dominant_gender": str(info.get("dominant_gender", "未知"))
                })

            # 保存elbow_plot到文件
            import time
            elbow_filename = f"elbow_{int(time.time())}.png"
            elbow_filepath = os.path.join('static', elbow_filename)
            if elbow_plot:
                import base64
                img_data = base64.b64decode(elbow_plot)
                with open(elbow_filepath, 'wb') as f:
                    f.write(img_data)

            final_result = {
                "step": "done",
                "message": "聚类完成",
                "data": {
                    "optimal_k": int(n_clusters),
                    "clusters": cluster_overview,
                    "elbow_plot_url": f"/static/{elbow_filename}" if elbow_plot else None
                }
            }
            yield f'data: {__import__("json").dumps(final_result, ensure_ascii=False)}\n\n'

        except Exception as e:
            print(f"[聚类] 错误: {e}")
            traceback.print_exc()
            yield f'data: {{"step": "error", "message": "聚类失败: {str(e)}"}}\n\n'

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    LLM分析接口（同步版本，直接执行AI分析）
    """
    if not MINIMAX_API_KEY:
        return jsonify({
            "code": 1,
            "message": "请在config.py中配置MiniMax API Key"
        })

    if system_state["clustering_result"] is None:
        return jsonify({"code": 1, "message": "请先进行聚类分析"})

    if system_state["analysis_done"]:
        return jsonify({"code": 1, "message": "已经分析过"})

    try:
        clustering_result = system_state["clustering_result"]
        cluster_features = clustering_result["cluster_features"]

        from recommendation import analyze_and_recommend_cluster

        for cluster_id, info in cluster_features.items():
            indicator_data = {}
            for key, data in info.get("indicator_data", {}).items():
                parts = key.rsplit("_", 1)
                indicator = parts[0]
                gender = parts[1] if len(parts) > 1 else ""
                if indicator not in indicator_data:
                    indicator_data[indicator] = {}
                indicator_data[indicator] = {
                    "mean": data.get("mean", 0),
                    "std": data.get("std", 0),
                    "gender": gender
                }

            if indicator_data:
                result = analyze_and_recommend_cluster(indicator_data, MINIMAX_API_KEY)
                if result:
                    info["cluster_type"] = result.get("cluster_type", "待评估")
                    info["description"] = result.get("description", "")
                    info["weak_features"] = result.get("weak_features", [])
                    info["strong_features"] = result.get("strong_features", [])
                    info["recommendations"] = result.get("recommendations", [])
                else:
                    info["cluster_type"] = "待评估"
                    info["description"] = "AI分析服务暂时不可用"
                    info["weak_features"] = []
                    info["strong_features"] = []
                    info["recommendations"] = []

        # 智能课程分配 - 解决扎堆问题
        total_students = sum(info.get("size", 0) for info in cluster_features.values())
        course_recommendations = {
            cid: info.get("recommendations", [])
            for cid, info in cluster_features.items()
        }
        from recommendation import allocate_courses_smartly
        allocation_result = allocate_courses_smartly(cluster_features, course_recommendations, total_students)

        # 用分配结果更新每个群体的recommendations
        for cluster_id, allocations in allocation_result.items():
            if cluster_id in cluster_features:
                cluster_features[cluster_id]["recommendations"] = allocations
                cluster_features[cluster_id]["allocation_details"] = allocations

        system_state["analysis_done"] = True

        cluster_overview = []
        for cid, info in cluster_features.items():
            cluster_overview.append({
                "cluster_id": cid,
                "label": info.get("label", f"C{cid+1}"),
                "ratio": info.get("ratio", 0),
                "size": info.get("size", 0),
                "cluster_type": info.get("cluster_type", "未知"),
                "description": info.get("description", ""),
                "weak_features": info.get("weak_features", []),
                "strong_features": info.get("strong_features", []),
                "recommendations": info.get("recommendations", [])
            })

        system_state["analysis_result"] = cluster_overview

        return jsonify({
            "code": 0,
            "data": {"clusters": cluster_overview}
        })

    except Exception as e:
        return jsonify({"code": 1, "message": f"分析失败: {str(e)}"})


@app.route('/api/analysis-status', methods=['GET'])
def analysis_status():
    """
    查询AI分析状态和结果
    """
    if system_state["clustering_result"] is None:
        return jsonify({"code": 1, "message": "请先进行聚类分析"})

    if system_state["analysis_done"]:
        return jsonify({
            "code": 0,
            "done": True,
            "data": {"clusters": convert_to_native_types(system_state["analysis_result"])}
        })
    else:
        return jsonify({
            "code": 0,
            "done": False,
            "message": "AI分析进行中..."
        })


@app.route('/api/start-analysis', methods=['POST'])
def start_analysis():
    """
    启动后台AI分析（非阻塞）
    """
    if not MINIMAX_API_KEY:
        return jsonify({"code": 1, "message": "请配置API Key"})

    if system_state["clustering_result"] is None:
        return jsonify({"code": 1, "message": "请先进行聚类"})

    if system_state["analysis_done"]:
        return jsonify({"code": 0, "message": "已经分析完成"})

    def run_analysis():
        """后台分析函数 - 一步完成"""
        import time
        import re
        try:
            clustering_result = system_state["clustering_result"]
            cluster_features = clustering_result["cluster_features"]

            from recommendation import call_minimax_api

            total_clusters = len(cluster_features)
            print(f"[AI分析开始] 共{total_clusters}个群体，一步完成")

            # 构建批量数据
            batch_data = {}
            for cid, info in cluster_features.items():
                batch_data[cid] = {
                    "label": info.get("label", f"C{cid+1}"),
                    "relative_data": info.get("relative_data", {})
                }

            total_start = time.time()

            # 一步分析：类型+优势+劣势
            print("[分析] 正在分析...")
            prompt = _build_one_step_prompt(batch_data)
            print(f"[分析] prompt长度: {len(prompt)}")

            resp = call_minimax_api(prompt, MINIMAX_API_KEY)
            if resp:
                print(f"[分析] 原始响应:\n{resp}\n---")
                parsed = _parse_one_step_response(resp.strip())
                print(f"[分析] 解析结果: {parsed}")

                for cid, info in cluster_features.items():
                    idx = list(cluster_features.keys()).index(cid)
                    # parsed 的 key 是 [C#] 中的编号（1,2,3...），需要 +1 匹配
                    llm_label = idx + 1
                    if llm_label in parsed:
                        p = parsed[llm_label]
                        info["raw_text"] = p.get("raw_text", "")
                        info["cluster_type"] = p.get("type", "待评估")
                        info["description"] = p.get("description", "")
                        info["strong_features"] = p.get("strong", [])
                        info["weak_features"] = p.get("weak", [])

                        # 直接使用解析好的推荐列表（解析函数已处理）
                        recommendations = p.get("recommendation", [])

                        # 确保正好2个推荐
                        if len(recommendations) < 2:
                            if len(recommendations) == 0:
                                recommendations.append({
                                    "course_name": "健美操",
                                    "type": "提升劣势",
                                    "reason": "全面提升身体素质，针对性改善体能短板"
                                })
                            recommendations.append({
                                "course_name": "田径",
                                "type": "巩固优势",
                                "reason": "基础体能训练，巩固现有优势"
                            })

                        info["recommendations"] = recommendations[:2]
                    else:
                        info["cluster_type"] = "待评估"
                        info["description"] = "AI分析服务暂时不可用"
                        info["strong_features"] = []
                        info["weak_features"] = []
                        info["recommendations"] = [
                            {"course_name": "健美操", "type": "提升劣势", "reason": "综合提升身体素质"},
                            {"course_name": "田径", "type": "巩固优势", "reason": "基础体能训练"},
                        ]
            else:
                print("[分析] 失败")
                for cid, info in cluster_features.items():
                    info["cluster_type"] = "待评估"
                    info["description"] = "AI分析服务暂时不可用"
                    info["strong_features"] = []
                    info["weak_features"] = []
                    info["recommendations"] = [
                        {"course_name": "健美操", "type": "提升劣势", "reason": "综合提升身体素质"},
                        {"course_name": "田径", "type": "巩固优势", "reason": "基础体能训练"},
                    ]

            total_time = time.time() - total_start
            print(f"[AI分析] 总耗时: {total_time:.1f}秒")

            # 标记完成并保存结果
            cluster_overview = []
            for cid, info in cluster_features.items():
                cluster_overview.append({
                    "cluster_id": cid,
                    "label": info.get("label", f"C{cid+1}"),
                    "ratio": info.get("ratio", 0),
                    "size": info.get("size", 0),
                    "raw_text": info.get("raw_text", ""),
                    "cluster_type": info.get("cluster_type", "未知"),
                    "description": info.get("description", ""),
                    "weak_features": info.get("weak_features", []),
                    "strong_features": info.get("strong_features", []),
                    "recommendations": info.get("recommendations", [])
                })

            system_state["analysis_done"] = True
            system_state["analysis_result"] = cluster_overview
            print(f"后台AI分析完成，共{len(cluster_overview)}个群体")

        except Exception as e:
            print(f"后台AI分析失败: {e}")
            import traceback
            traceback.print_exc()

    # 启动后台线程
    thread = threading.Thread(target=run_analysis)
    thread.daemon = True
    thread.start()

    return jsonify({"code": 0, "message": "已开始后台分析"})


def _build_one_step_prompt(batch_data):
    """构建一步分析prompt"""
    import re

    # 构建简洁的数据展示
    parts = []
    for cid, data in batch_data.items():
        label = data.get("label", f"C{cid+1}")
        rel = data.get("relative_data", {})

        # 简化为：指标名:相对值（不带性别）
        detail_parts = []
        for key, val in rel.items():
            indicator = key[:3]  # 只取前3个字
            rel_val = val.get("relative", 1.0)

            # 用箭头表示高低
            if rel_val >= 1.1:
                arrow = "↑"
            elif rel_val <= 0.9:
                arrow = "↓"
            else:
                arrow = "→"

            detail_parts.append(f"{indicator}{arrow}{rel_val:.2f}")

        detail_str = " ".join(detail_parts)
        parts.append(f"{label}|{detail_str}")

    course_list = "田径、篮球、足球、排球、乒乓球、健美操、跆拳道、散打、游泳、养生、太极拳、网球、羽毛球、中国式摔跤、拳击、健身健美"

    # 课程与能力对应关系，用于引导 LLM 精准推荐
    course_ability_map = """
    【课程-能力对应表】
    - 游泳/田径(中长跑)/健美操 → 针对心肺耐力差
    - 健身健美/拳击/跆拳道/中国式摔跤 → 针对力量差
    - 田径(短跑)/篮球/羽毛球/排球 → 针对速度/爆发力差
    - 健美操/太极拳/瑜伽/养生 → 针对柔韧性差
    - 乒乓球/羽毛球/网球 → 针对反应速度/协调性
    """

    return f"""分析学生群体体测数据，为每个群体起一个运动项目风格的名称，并精准推荐课程。

【数据：群体|指标相对值】（↑高于平均，↓低于平均，→接近平均）
{" | ".join(parts)}

【课程库】{course_list}
{course_ability_map}

【输出格式】（严格按此格式，每群体一段，共{len(batch_data)}段）
[C1]
类型：[用运动项目命名，如"径赛选手""举重选手""体操选手""游泳选手""马拉松选手""铁人选手"等，要形象有趣，体现该群体最突出的体能特点]
特点：[完整的一句话体质描述，简洁生动，不要截断]
优势：[列出该群体最突出的1-2个能力，如"耐力出众""柔韧性极佳"]
短板：[列出该群体最需提升的1-2个能力，如"力量偏弱""速度有待提高"]
推荐：
- 提升劣势：[推荐课程名称] - [推荐理由，要说明这门课如何改善用户的短板能力，40字以内]
- 巩固优势：[推荐课程名称] - [推荐理由，要说明这门课如何进一步发挥用户的优势，40字以内]

【重要规则】
1. 类型名必须用运动项目比喻法，优先选择最能代表群体优势的运动项目名
2. 推荐必须正好2个课程：第一个"提升劣势"，第二个"巩固优势"
3. 理由要具体说明课程与能力提升的关联，让用户明白为什么推荐这门课
4. 同一群体的两个推荐课程必须不同，避免推荐相同课程
5. 必须为全部{len(batch_data)}个群体输出，不能跳过任何编号"""


def _parse_one_step_response(text):
    """
    解析一步响应 - 匹配新的格式
    格式：
    [C1]
    类型：[类型名]
    特点：[一句话描述]
    优势：[优势]
    短板：[短板]
    推荐：[课程](置信度)理由/[课程](置信度)理由
    """
    import re
    results = {}

    # 匹配 [C1] 或 [C2] 等段落
    pattern = r'\[C(\d+)\](.*?)(?=\[C\d+\]|$)'
    matches = re.findall(pattern, text, re.DOTALL)

    print(f"[解析] 找到 {len(matches)} 个 [C#] 段落")

    for cid_str, content in matches:
        cid_num = int(cid_str)
        content = content.strip()

        # 保存原始文本
        raw_text = f"[C{cid_str}]\n{content}"

        # 提取类型 - 格式：类型：[xxx]
        type_m = re.search(r'类型[：:]\s*(.+?)(?:\n|特点)', content)
        cluster_type = type_m.group(1).strip() if type_m else "待评估"

        # 提取特点 - 格式：特点：[xxx]
        desc_m = re.search(r'特点[：:]\s*(.+?)(?:\n|优势|短板)', content)
        description = desc_m.group(1).strip() if desc_m else ""

        # 提取优势 - 格式：优势：[xxx]
        strong_m = re.search(r'优势[：:]\s*(.+?)(?:\n|短板|推荐)', content)
        strong_features = []
        if strong_m:
            strong_str = strong_m.group(1).strip()
            for s in re.split(r'[,，、/]', strong_str):
                s = s.strip()
                if s and s not in ['无', '暂无', '无明显优势', '无明显短板']:
                    strong_features.append(s)

        # 提取短板 - 格式：短板：[xxx]
        weak_m = re.search(r'短板[：:]\s*(.+?)(?:\n|推荐)', content)
        weak_features = []
        if weak_m:
            weak_str = weak_m.group(1).strip()
            for s in re.split(r'[,，、/]', weak_str):
                s = s.strip()
                if s and s not in ['无', '暂无', '无明显优势', '无明显短板']:
                    weak_features.append(s)

        # 提取推荐 - 新格式：推荐：\n- 提升劣势：[课程] - [理由]\n- 巩固优势：[课程] - [理由]
        recommendations = []
        rec_m = re.search(r'推荐[：:]\s*(.+?)(?=\[C\d+\]|$)', content, re.DOTALL)
        if rec_m:
            rec_text = rec_m.group(1).strip()
            # 匹配 "- 提升劣势：[课程名] - [理由]" 格式
            improve_pattern = r'[-–]\s*提升劣势[：:]\s*([^]-]+?)\s*[-–]\s*(.+?)(?=\n[-–]|$$)'
            improve_match = re.search(improve_pattern, rec_text, re.DOTALL)
            if improve_match:
                course_name = improve_match.group(1).strip()
                reason = improve_match.group(2).strip()
                if course_name and reason:
                    recommendations.append({
                        "course_name": course_name,
                        "type": "提升劣势",
                        "reason": reason
                    })

            # 匹配 "- 巩固优势：[课程名] - [理由]" 格式
            strengthen_pattern = r'[-–]\s*巩固优势[：:]\s*([^]-]+?)\s*[-–]\s*(.+?)(?=\n[-–]|$$)'
            strengthen_match = re.search(strengthen_pattern, rec_text, re.DOTALL)
            if strengthen_match:
                course_name = strengthen_match.group(1).strip()
                reason = strengthen_match.group(2).strip()
                if course_name and reason:
                    recommendations.append({
                        "course_name": course_name,
                        "type": "巩固优势",
                        "reason": reason
                    })

        # 如果解析失败，尝试旧格式兼容
        if not recommendations:
            rec_pattern = r'([^()（）\n/]+?)\s*\((\d+)\)\s*([^/\n]+)'
            rec_matches = re.findall(rec_pattern, rec_text or '')
            for match in rec_matches:
                course_name = match[0].strip()
                try:
                    score = int(match[1])
                except:
                    score = 80
                reason = match[2].strip()
                if course_name and reason:
                    recommendations.append({
                        "course_name": course_name,
                        "score": score,
                        "reason": reason
                    })

        results[cid_num] = {
            "raw_text": raw_text,
            "type": cluster_type,
            "description": description,
            "strong": strong_features,
            "weak": weak_features,
            "recommendation": recommendations
        }

    print(f"[解析] 成功解析 {len(results)} 个群体")
    for cid, data in results.items():
        print(f"  [C{cid}] {data['type']} | 推荐: {len(data['recommendation'])}门")
    return results


def _generate_recommendations(weak_features, strong_features):
    """
    基于薄弱项和优势项生成推荐课程
    """
    # 课程映射：能力 -> 推荐课程
    COURSE_MAP = {
        "心肺耐力": [
            {"name": "有氧健身操", "score": 95, "reason": "有效提升心肺功能"},
            {"name": "游泳", "score": 90, "reason": "全身有氧运动，增强心肺"},
        ],
        "力量": [
            {"name": "力量训练", "score": 95, "reason": "增强肌肉力量"},
            {"name": "引体向上专项", "score": 85, "reason": "针对性提升上肢力量"},
        ],
        "速度": [
            {"name": "短跑训练", "score": 95, "reason": "提升速度素质"},
            {"name": "敏捷性训练", "score": 85, "reason": "改善反应速度和位移速度"},
        ],
        "爆发力": [
            {"name": "立定跳远专项", "score": 95, "reason": "增强下肢爆发力"},
            {"name": "跳绳", "score": 85, "reason": "提升弹跳力和协调性"},
        ],
        "柔韧性": [
            {"name": "瑜伽", "score": 95, "reason": "全面提升柔韧性"},
            {"name": "拉伸训练", "score": 90, "reason": "专门改善关节活动度"},
        ],
        "耐力": [
            {"name": "中长跑", "score": 95, "reason": "增强有氧耐力"},
            {"name": "有氧健身操", "score": 85, "reason": "持续运动提升耐力水平"},
        ],
    }

    recommendations = []
    seen = set()

    # 优先推荐改善薄弱项的课程
    for weak in weak_features:
        if weak in COURSE_MAP:
            for course in COURSE_MAP[weak]:
                if course["name"] not in seen:
                    recommendations.append({
                        "course_name": course["name"],
                        "score": course["score"],
                        "reason": course["reason"]
                    })
                    seen.add(course["name"])

    # 如果薄弱项课程不够3个，补充优势项维持课程
    if len(recommendations) < 3:
        for strong in strong_features:
            if strong in COURSE_MAP and len(recommendations) < 3:
                for course in COURSE_MAP[strong]:
                    if course["name"] not in seen:
                        recommendations.append({
                            "course_name": course["name"],
                            "score": course["score"] - 10,  # 优势项课程稍低优先级
                            "reason": f"巩固{strong}优势"
                        })
                        seen.add(course["name"])

    return recommendations[:3]  # 最多返回3个


@app.route('/api/recommend/<int:cluster_id>', methods=['GET'])
def recommend(cluster_id):
    """
    获取指定聚类的课程推荐（由LLM生成）
    """
    if not MINIMAX_API_KEY:
        return jsonify({
            "code": 1,
            "message": "请在config.py中配置MiniMax API Key后再获取推荐"
        })

    if system_state["clustering_result"] is None:
        return jsonify({"code": 1, "message": "请先进行聚类分析"})

    try:
        clustering_result = system_state["clustering_result"]
        cluster_info = clustering_result["cluster_features"].get(cluster_id)

        if cluster_info is None:
            return jsonify({"code": 1, "message": "聚类ID不存在"})

        # 构建群体分析结果
        cluster_analysis = {
            "cluster_type": cluster_info.get("cluster_type", "未知类型"),
            "description": cluster_info.get("description", ""),
            "weak_features": cluster_info.get("weak_features", []),
            "strong_features": cluster_info.get("strong_features", []),
            "ratio": cluster_info.get("ratio", 0)
        }

        # 调用LLM生成推荐
        recommendations = recommend_courses_with_llm(cluster_analysis, MINIMAX_API_KEY)

        if not recommendations:
            return jsonify({
                "code": 1,
                "message": "LLM调用失败，请检查API Key是否正确"
            })

        return jsonify({
            "code": 0,
            "data": {
                "cluster_info": {
                    "cluster_id": cluster_id,
                    "label": cluster_info.get("label", f"C{cluster_id+1}"),
                    "ratio": cluster_info.get("ratio", 0),
                    "cluster_type": cluster_info.get("cluster_type", "未知类型"),
                    "description": cluster_info.get("description", ""),
                    "weak_features": cluster_info.get("weak_features", []),
                    "strong_features": cluster_info.get("strong_features", [])
                },
                "recommendations": recommendations
            }
        })

    except Exception as e:
        return jsonify({"code": 1, "message": f"推荐失败: {str(e)}"})


@app.route('/api/course-decision', methods=['GET'])
def course_decision():
    """
    获取课程配置决策建议（由LLM生成）
    """
    if not MINIMAX_API_KEY:
        return jsonify({
            "code": 1,
            "message": "请在config.py中配置MiniMax API Key后再获取建议"
        })

    if system_state["clustering_result"] is None:
        return jsonify({"code": 1, "message": "请先进行聚类分析"})

    try:
        clustering_result = system_state["clustering_result"]
        cluster_features_list = list(clustering_result["cluster_features"].values())

        # 统计现有课程分布
        course_distribution = {}
        for course in SPORT_COURSES:
            cat = course["category"]
            course_distribution[cat] = course_distribution.get(cat, 0) + 1

        # 构建群体分析列表
        cluster_analyses = []
        for info in cluster_features_list:
            cluster_analyses.append({
                "cluster_type": info.get("cluster_type", "未知类型"),
                "description": info.get("description", ""),
                "weak_features": info.get("weak_features", []),
                "strong_features": info.get("strong_features", []),
                "ratio": info.get("ratio", 0)
            })

        # 调用LLM生成决策建议
        llm_suggestion = generate_course_decision_with_llm(
            cluster_analyses, course_distribution, MINIMAX_API_KEY
        )

        return jsonify({
            "code": 0,
            "data": {
                "current_distribution": course_distribution,
                "llm_suggestion": llm_suggestion
            }
        })

    except Exception as e:
        return jsonify({"code": 1, "message": f"决策建议生成失败: {str(e)}"})


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    聊天接口（非流式）
    """
    print(f"[聊天] clustering_result状态: {system_state['clustering_result'] is not None}")

    if not MINIMAX_API_KEY:
        return jsonify({"code": 1, "message": "请在config.py中配置API Key"})

    if system_state["clustering_result"] is None:
        print(f"[聊天] 错误: clustering_result is None")
        return jsonify({"code": 1, "message": "请先进行聚类分析"})

    try:
        data = request.get_json()
        cluster_id = data.get('cluster_id')
        message = data.get('message', '')

        clustering_result = system_state["clustering_result"]
        current_cluster_info = clustering_result["cluster_features"].get(cluster_id, {})

        # 获取聊天历史
        chat_history = system_state.get("chat_history", [])
        history_str = ""
        if chat_history:
            history_lines = []
            for h in chat_history[-6:]:  # 只保留最近6条
                history_lines.append(f"用户：{h['user']}")
                history_lines.append(f"助手：{h['assistant']}")
            history_str = "\n".join(history_lines)

        # 构建提示词
        weak = ', '.join(current_cluster_info.get('weak_features', []))
        strong = ', '.join(current_cluster_info.get('strong_features', []))

        prompt_parts = ["""你是体育课程推荐专家。用户属于以下群体，请根据群体特征和对话历史回答用户问题。"""]

        if history_str:
            prompt_parts.append(f"\n对话历史：\n{history_str}")

        prompt_parts.append(f"""
群体信息：
- 群体名称：{current_cluster_info.get('label', '未知')}
- 类型：{current_cluster_info.get('cluster_type', '未知')}
- 描述：{current_cluster_info.get('description', '')}
- 薄弱项（需要重点提升的能力）：{weak}
- 优势项（相对较强的能力）：{strong}

用户问题：{message}

重要理解规则：
- 当用户问"什么课对我帮助最小"或"不推荐什么课"时：
  → 是指那些主要依赖用户薄弱能力的课程，用户很难在短期内看到进步，容易受挫
  → 应该回答：以你的薄弱项（{weak}）为主的课程对你帮助最小，因为这些课程需要你目前最弱的能力
- 当用户问"推荐什么课"时：
  → 应该优先推荐能改善薄弱项的课程，帮助用户针对性提升

请用专业、友好的语气回答。可以根据用户具体情况（如有无伤病等）给出个性化建议。

回答要求：
1. 简洁专业，控制在100字以内
2. 如涉及课程推荐，要给出具体课程名称
3. 如用户有特殊身体情况（如膝盖不好），需在建议中考虑
""")

        prompt = ''.join(prompt_parts)

        # 调用非流式API
        response = call_minimax_api(prompt, MINIMAX_API_KEY)

        if not response:
            return jsonify({"code": 1, "message": "AI服务暂时不可用，请稍后重试"})

        # 保存对话历史
        if "chat_history" not in system_state:
            system_state["chat_history"] = []
        system_state["chat_history"].append({
            "user": message,
            "assistant": response
        })

        return jsonify({
            "code": 0,
            "data": {
                "response": response
            }
        })

    except Exception as e:
        return jsonify({"code": 1, "message": f"聊天失败: {str(e)}"})


@app.route('/api/student-cluster/<int:student_idx>', methods=['GET'])
def get_student_cluster(student_idx):
    """
    获取指定学生的聚类标签
    """
    if system_state["current_labels"] is None:
        return jsonify({"code": 1, "message": "请先进行聚类分析"})

    try:
        labels = system_state["current_labels"]
        if student_idx >= len(labels):
            return jsonify({"code": 1, "message": "学生索引超出范围"})

        cluster_id = labels[student_idx]
        clustering_result = system_state["clustering_result"]
        cluster_info = clustering_result["cluster_features"].get(cluster_id)

        return jsonify({
            "code": 0,
            "data": {
                "student_index": student_idx,
                "cluster_id": cluster_id,
                "cluster_label": cluster_info.get("label", f"C{cluster_id+1}") if cluster_info else f"C{cluster_id+1}",
                "cluster_type": cluster_info.get("cluster_type", "未知类型") if cluster_info else "未知类型"
            }
        })

    except Exception as e:
        return jsonify({"code": 1, "message": f"获取失败: {str(e)}"})


if __name__ == '__main__':
    print("=" * 50)
    print("体测数据驱动的大语言模型体育课程个性化推荐系统")
    print("=" * 50)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
