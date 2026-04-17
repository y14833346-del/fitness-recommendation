// ========================================
// Global State
// ========================================
const state = {
    currentStep: 1,
    uploadedFile: null,
    clusters: [],
    clustersOriginal: [],
    currentCluster: null,
    currentFilter: '',
    currentSort: 'size-desc',
    chatMessages: []
};

// ===========================================
// Initialization
// ========================================
document.addEventListener('DOMContentLoaded', async () => {
    // 刷新页面时清除后端旧状态
    try {
        await fetch('/api/reset', { method: 'POST' });
    } catch (e) {}
    initUploadZone();
    initChat();
});

function initUploadZone() {
    const zone = document.getElementById('upload-zone');
    const input = document.getElementById('file-input');

    zone.addEventListener('click', () => {
        input.click();
    });

    input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
}

function initChat() {
    const chatInput = document.getElementById('chat-input');
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChat();
        }
    });
}

// ========================================
// File Handling
// ========================================
function handleFile(file) {
    const validExts = ['.csv', '.xlsx', '.xls'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    if (!validExts.includes(ext)) {
        showToast('请上传 CSV 或 Excel 文件', 'error');
        return;
    }

    state.uploadedFile = file;

    document.getElementById('upload-zone').style.display = 'none';
    document.getElementById('file-preview').style.display = 'flex';
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = formatFileSize(file.size);
    document.getElementById('btn-preprocess').disabled = false;

    updateProgress(1);
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function clearFile() {
    state.uploadedFile = null;
    state.clusters = [];
    state.currentCluster = null;

    document.getElementById('file-input').value = '';
    document.getElementById('upload-zone').style.display = 'block';
    document.getElementById('file-preview').style.display = 'none';
    document.getElementById('btn-preprocess').disabled = true;

    updateProgress(1);
    showSection('section-upload');
}

function resetUpload() {
    clearFile();
}

// ========================================
// Progress & Navigation
// ========================================
function updateProgress(step) {
    state.currentStep = step;

    document.querySelectorAll('.step').forEach((el, idx) => {
        el.classList.remove('active', 'completed');
        if (idx + 1 < step) {
            el.classList.add('completed');
        } else if (idx + 1 === step) {
            el.classList.add('active');
        }
    });

    // Update progress bar visibility
    const progressBar = document.getElementById('progress-bar');
    if (step > 1) {
        progressBar.style.display = 'block';
    } else {
        progressBar.style.display = 'none';
    }

    // Update progress text and fill
    const progressText = document.getElementById('progress-text');
    const progressFill = document.getElementById('progress-fill');
    const texts = ['', '上传数据', '数据预览', '聚类分析', 'AI分析+课程咨询'];
    const percentages = [0, 20, 40, 60, 100];
    progressText.textContent = texts[step] || '';
    progressFill.style.width = percentages[step] + '%';
}

function showSection(id) {
    console.log('[showSection] 切换到:', id);
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(id).classList.add('active');
    console.log('[showSection] 切换完成');
}

// ========================================
// Data Preprocessing
// ========================================
async function preprocessData() {
    if (!state.uploadedFile) {
        showToast('请先上传文件', 'error');
        return;
    }

    const btn = document.getElementById('btn-preprocess');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-small" style="margin-right:8px;"></span>处理中...';

    try {
        const formData = new FormData();
        formData.append('file', state.uploadedFile);

        const response = await fetch('/api/preprocess', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.code === 0) {
            state.preprocessResult = result.data;

            document.getElementById('stat-samples').textContent = result.data.total_samples;
            document.getElementById('stat-indicators').textContent = result.data.valid_columns.length;

            renderPreviewTable(result.data);

            updateProgress(2);
            showSection('section-preview');

            showToast('预处理成功', 'success');
        } else {
            showToast(result.message, 'error');
        }
    } catch (err) {
        showToast('预处理失败: ' + err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '开始预处理 <span class="btn-icon-right">→</span>';
    }
}

function renderPreviewTable(data) {
    const table = document.getElementById('preview-table');
    const thead = table.querySelector('thead');
    const tbody = table.querySelector('tbody');

    if (data.sample_preview && data.sample_preview.length > 0) {
        const columns = Object.keys(data.sample_preview[0]);
        thead.innerHTML = '<tr>' + columns.map(col => `<th>${col}</th>`).join('') + '</tr>';
        tbody.innerHTML = data.sample_preview.map(row =>
            '<tr>' + columns.map(col => `<td>${row[col] ?? ''}</td>`).join('') + '</tr>'
        ).join('');
    }
}

// ========================================
// Clustering with Real-time Progress
// ========================================
async function startClustering() {
    updateProgress(3);
    showSection('section-clustering');

    // 显示加载状态
    document.getElementById('clustering-loading').style.display = 'flex';
    document.getElementById('clustering-results').style.display = 'none';

    try {
        const response = await fetch('/api/cluster', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalData = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // 处理缓冲区的每一行
            let lines = buffer.split('\n');
            buffer = lines.pop() || ''; // 保留未完成的行

            for (const line of lines) {
                const trimmedLine = line.trim();
                if (trimmedLine.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(trimmedLine.slice(6));

                        if (data.step === 'error') {
                            showToast(data.message, 'error');
                            showSection('section-preview');
                            updateProgress(2);
                            return;
                        }

                        if (data.step === 'done') {
                            finalData = data.data;
                        } else {
                            updateLoadingStatus(data.message, data.progress !== undefined ? `进度 ${data.progress}%` : '');
                        }
                    } catch (e) {
                        console.error('JSON解析失败:', e, 'raw:', trimmedLine);
                    }
                }
            }
        }

        // 处理缓冲区中剩余的数据
        if (buffer.trim()) {
            const trimmedLine = buffer.trim();
            if (trimmedLine.startsWith('data: ')) {
                try {
                    const data = JSON.parse(trimmedLine.slice(6));
                    if (data.step === 'done') {
                        finalData = data.data;
                    }
                } catch (e) {
                    console.error('JSON解析失败(尾部):', e, 'raw:', trimmedLine);
                }
            }
        }

        if (finalData) {
            state.clusters = finalData.clusters;

            // 隐藏加载状态，显示结果
            document.getElementById('clustering-loading').style.display = 'none';
            document.getElementById('clustering-results').style.display = 'flex';

            document.getElementById('result-k').textContent = finalData.optimal_k;

            // 显示聚类图表
            if (finalData.elbow_plot_url) {
                document.getElementById('elbow-plot').src = finalData.elbow_plot_url;
                document.getElementById('chart-section').style.display = 'block';
            }

            // 渲染聚类结果（只显示群体+占比）
            renderClusteringTable(finalData.clusters);

            // 后台开始AI分析
            startBackgroundAnalysis();

        } else {
            showToast('聚类失败：未收到结果', 'error');
            showSection('section-preview');
            updateProgress(2);
        }
    } catch (err) {
        showToast('聚类失败: ' + err.message, 'error');
        showSection('section-preview');
        updateProgress(2);
    }
}

let analysisCheckTimer = null;

async function startBackgroundAnalysis() {
    // 启动后台AI分析
    try {
        const response = await fetch('/api/start-analysis', {
            method: 'POST'
        });
        const result = await response.json();
        if (result.code === 0) {
            console.log('后台AI分析已启动');
        }
    } catch (err) {
        console.log('启动后台分析失败:', err);
    }
}

let isCheckingAnalysis = false;

async function goToAIResults() {
    // 防止重复执行
    if (isCheckingAnalysis) {
        console.log('[goToAIResults] 正在检查中，跳过');
        return;
    }
    isCheckingAnalysis = true;
    console.log('[goToAIResults] ===== 开始 =====');

    // 前往AI分析+课程咨询页面，检查分析状态
    try {
        console.log('[goToAIResults] 开始检查分析状态');
        const response = await fetch('/api/analysis-status');
        const result = await response.json();
        console.log('[goToAIResults] 状态结果:', result);
        console.log('[goToAIResults] 判断: code=%d, done=%s, hasData=%s', result.code, result.done, !!result.data);

        if (result.code === 0 && result.done && result.data) {
            // AI分析已完成，显示结果
            console.log('[goToAIResults] 分析已完成，显示结果');
            state.clusters = result.data.clusters;
            renderAIResultsTable(result.data.clusters);
            renderTypeChips(result.data.clusters);

            if (result.data.clusters.length > 0) {
                selectClusterType(result.data.clusters[0].cluster_id);
            }

            updateProgress(4);
            showSection('section-chat');
            showToast('AI分析完成', 'success');
        } else if (result.code !== 0) {
            // 后端返回错误
            console.log('[goToAIResults] 后端错误:', result.message);
            showToast(result.message || '查询失败', 'error');
            isCheckingAnalysis = false;
        } else {
            console.log('[goToAIResults] 分析进行中，显示等待状态');
            // AI分析还在进行中，显示加载状态
            updateProgress(4);
            showSection('section-clustering');
            document.getElementById('clustering-loading').style.display = 'flex';
            document.getElementById('clustering-results').style.display = 'none';
            updateLoadingStatus('正在等待AI分析完成...', '请稍候');

            // 停止之前的检查定时器
            if (analysisCheckTimer) {
                clearInterval(analysisCheckTimer);
            }

            // 轮询超时：60秒后自动停止
            const timeoutId = setTimeout(() => {
                if (analysisCheckTimer) {
                    clearInterval(analysisCheckTimer);
                    analysisCheckTimer = null;
                    showToast('AI分析超时，请重试', 'error');
                    isCheckingAnalysis = false;
                }
            }, 60000);

            // 定期检查分析状态
            analysisCheckTimer = setInterval(async () => {
                console.log('[轮询] 检查分析状态');
                const statusResponse = await fetch('/api/analysis-status');
                const statusResult = await statusResponse.json();
                console.log('[轮询] 状态结果:', statusResult);
                console.log('[轮询] 判断: code=%d, done=%s, hasData=%s', statusResult.code, statusResult.done, !!statusResult.data);

                if (statusResult.code === 0 && statusResult.done && statusResult.data) {
                    console.log('[轮询] 分析完成，停止轮询，显示结果');
                    clearInterval(analysisCheckTimer);
                    analysisCheckTimer = null;
                    clearTimeout(timeoutId);

                    // AI分析完成，显示结果
                    state.clusters = statusResult.data.clusters;
                    renderAIResultsTable(statusResult.data.clusters);
                    renderTypeChips(statusResult.data.clusters);

                    if (statusResult.data.clusters.length > 0) {
                        console.log('[显示结果] 调用selectClusterType, id:', statusResult.data.clusters[0].cluster_id);
                        selectClusterType(statusResult.data.clusters[0].cluster_id);
                    }

                    console.log('[显示结果] 调用updateProgress和showSection');
                    updateProgress(4);
                    showSection('section-chat');
                    showToast('AI分析完成', 'success');
                    console.log('[显示结果] 完成');
                    isCheckingAnalysis = false;
                } else if (statusResult.code !== 0) {
                    // 后端返回错误，停止轮询
                    console.log('[轮询] 后端错误:', statusResult.message);
                    clearInterval(analysisCheckTimer);
                    analysisCheckTimer = null;
                    clearTimeout(timeoutId);
                    showToast(statusResult.message || '分析失败', 'error');
                    isCheckingAnalysis = false;
                }
            }, 1500);
        }
    } catch (err) {
        showToast('检查分析状态失败', 'error');
        updateProgress(4);
        showSection('section-chat');
        isCheckingAnalysis = false;
    }
}

function updateLoadingStatus(title, status) {
    document.getElementById('loading-title').textContent = title;
    document.getElementById('loading-status').textContent = status;
}

// 聚类结果表格（只显示群体+占比）
function renderClusteringTable(clusters) {
    const tbody = document.getElementById('result-tbody');
    tbody.innerHTML = clusters.map(cluster => {
        return `<tr data-cluster-id="${cluster.cluster_id}">
            <td>
                <span class="cluster-name">${cluster.label}</span>
            </td>
            <td>
                <span>${cluster.ratio}%</span>
                <span style="color:var(--gray-400);font-size:0.8rem;margin-left:8px;">(${cluster.size}人)</span>
            </td>
        </tr>`;
    }).join('');
}

// AI分析结果卡片（显示完整信息）
function renderAIResultsTable(clusters) {
    console.log('[renderAIResultsTable] 开始渲染, clusters数量:', clusters.length);
    renderClusterCards(clusters);
    console.log('[renderAIResultsTable] 渲染完成');
}

function renderClusterCards(clusters) {
    const container = document.getElementById('ai-cards-container');
    container.innerHTML = clusters.map(cluster => {
        const displayType = cluster.cluster_type || '待评估';
        const label = cluster.label || `C${cluster.cluster_id + 1}`;
        const size = cluster.size || 0;
        const ratio = cluster.ratio || 0;

        // 优势/劣势标签
        const weakTags = (cluster.weak_features || []).map(w =>
            `<span class="ability-tag weak"><span class="arrow">↓</span>${w}</span>`
        ).join('');
        const strongTags = (cluster.strong_features || []).map(s =>
            `<span class="ability-tag strong"><span class="arrow">↑</span>${s}</span>`
        ).join('');

        // 生成一句话总结
        let summary = cluster.description || '暂无详细描述';
        if (summary.length > 80) {
            summary = summary.substring(0, 80) + '...';
        }

        // 课程推荐 - 新格式：2个课程（提升劣势 + 巩固优势）
        let coursesHtml = '';
        if (cluster.recommendations && cluster.recommendations.length > 0) {
            const recs = cluster.recommendations.slice(0, 2);
            coursesHtml = recs.map((rec, i) => {
                const typeLabel = rec.type || (i === 0 ? '提升劣势' : '巩固优势');
                const typeClass = i === 0 ? 'improve' : 'strengthen';
                return `
                <div class="course-rec ${typeClass}">
                    <div class="course-type-badge">${typeLabel}</div>
                    <div class="course-rec-content">
                        <div class="course-rec-name">${rec.course_name}</div>
                        <div class="course-reason">${rec.reason || ''}</div>
                    </div>
                </div>`;
            }).join('');
        } else {
            coursesHtml = '<div style="color:var(--gray-400);font-size:0.85rem;">暂无推荐</div>';
        }

        // 生成雷达图数据（如果有相对数据）
        const radarData = generateRadarData(cluster);

        return `<div class="ai-card" data-cluster-id="${cluster.cluster_id}" onclick="selectClusterType(${cluster.cluster_id})">
            <div class="ai-card-header">
                <div class="ai-card-label">
                    <span class="ai-card-name">${label}</span>
                    <span class="ai-card-badge">${ratio}%</span>
                </div>
                <span class="ai-card-size">${size}人</span>
            </div>
            <div class="ai-card-type">${displayType}</div>
            <div class="ai-card-summary">${summary}</div>
            <div class="ability-tags">
                ${weakTags}
                ${strongTags}
            </div>
            ${radarData ? `<canvas class="ai-card-radar" id="radar-${cluster.cluster_id}" data-values="${radarData.values}" data-labels="${radarData.labels}"></canvas>` : ''}
            <div class="ai-card-courses">
                ${coursesHtml}
            </div>
        </div>`;
    }).join('');

    // 渲染雷达图
    setTimeout(renderRadarCharts, 50);
}

function generateRadarData(cluster) {
    // 从 relative_data 构建雷达图数据
    const rel = cluster.relative_data || {};
    const labels = [];
    const values = [];

    // 指标顺序
    const order = ['肺活量', '50米跑', '立定跳远', '坐位体前屈', '耐力跑', '力量'];
    const labelMap = {
        '肺活量': '心肺',
        '50米跑': '速度',
        '立定跳远': '爆发',
        '坐位体前屈': '柔韧',
        '耐力跑': '耐力',
        '力量': '力量'
    };

    order.forEach(key => {
        // 查找匹配的key
        for (const [relKey, relVal] of Object.entries(rel)) {
            if (relKey.startsWith(key)) {
                labels.push(labelMap[key] || key.substring(0, 2));
                let val = relVal.relative || 1.0;
                // 50米跑和耐力跑是越小越好，需要反转
                if (key === '50米跑' || key === '耐力跑') {
                    val = 2 - val; // 反转：差的值变成高的显示
                }
                values.push(Math.max(0, Math.min(2, val)));
                break;
            }
        }
    });

    if (labels.length === 0) return null;
    return { labels, values };
}

function renderRadarCharts() {
    document.querySelectorAll('.ai-card-radar').forEach(canvas => {
        const ctx = canvas.getContext('2d');
        const valuesStr = canvas.dataset.values;
        const labelsStr = canvas.dataset.labels;

        if (!valuesStr || !labelsStr) return;

        const values = valuesStr.split(',').map(parseFloat);
        const labels = labelsStr.split(',');

        new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: 'rgba(99, 102, 241, 0.2)',
                    borderColor: 'rgba(99, 102, 241, 0.8)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(99, 102, 241, 1)',
                    pointRadius: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        min: 0,
                        max: 2,
                        ticks: { display: false },
                        grid: { color: 'rgba(0,0,0,0.1)' },
                        angleLines: { color: 'rgba(0,0,0,0.1)' }
                    }
                }
            }
        });
    });
}

function renderResultsTable(clusters, showAIAnalysis = false) {
    const tbody = document.getElementById('result-tbody');
    tbody.innerHTML = clusters.map(cluster => {
        // Build features HTML
        const weakFeatures = (cluster.weak_features || []).map(f => `<span class="feature-tag weak">${f}</span>`).join('');
        const strongFeatures = (cluster.strong_features || []).map(f => `<span class="feature-tag strong">${f}</span>`).join('');
        const featuresHtml = weakFeatures + strongFeatures;

        // 如果不显示AI分析内容，只显示聚类基本信息
        if (!showAIAnalysis) {
            return `<tr data-cluster-id="${cluster.cluster_id}">
                <td>
                    <span class="cluster-name">${cluster.label}</span>
                    <div class="cluster-ratio" style="font-size:0.8rem;color:var(--gray-500);margin-top:4px;">占比 ${cluster.ratio}%</div>
                </td>
                <td colspan="3">
                    <span style="color:var(--gray-400);font-size:0.85rem;">点击"查看AI分析"获取详细信息</span>
                </td>
            </tr>`;
        }

        // 显示类型：待评估、未知、或实际类型
        const displayType = cluster.cluster_type || '待评估';
        const typeClass = (cluster.cluster_type && cluster.cluster_type !== '待评估') ? 'cluster-type' : 'cluster-type pending';

        // 如果有预加载的推荐，直接显示
        let recommendationsHtml = '';
        if (cluster.recommendations && cluster.recommendations.length > 0) {
            recommendationsHtml = cluster.recommendations.slice(0, 3).map(rec => {
                return `<div class="recommend-item">
                    <span class="recommend-course">${rec.course_name}</span>
                    <span class="recommend-score">${rec.score || 0}%</span>
                    <div class="recommend-reason">${rec.reason || ''}</div>
                </div>`;
            }).join('');
        } else if (!cluster.cluster_type) {
            // 还没有分析结果，显示加载状态
            recommendationsHtml = '<span class="spinner-small" style="display:inline-block;"></span> 分析中...';
        } else {
            // 有类型但没有推荐（分析刚完成但推荐未生成）
            recommendationsHtml = '<div style="color:var(--gray-400);font-size:0.8rem;">暂无推荐</div>';
        }

        return `<tr data-cluster-id="${cluster.cluster_id}">
            <td>
                <span class="cluster-name">${cluster.label}</span>
                <div class="cluster-ratio" style="font-size:0.8rem;color:var(--gray-500);margin-top:4px;">占比 ${cluster.ratio}%</div>
            </td>
            <td><span class="${typeClass}">${displayType}</span></td>
            <td>
                <div class="cluster-desc">${cluster.description || '等待AI分析...'}</div>
                <div style="margin-top:8px;">${featuresHtml || '<span style="color:var(--gray-400);font-size:0.8rem;">等待AI分析...</span>'}</div>
            </td>
            <td>
                <div class="recommend-cell" id="rec-cell-${cluster.cluster_id}">
                    ${recommendationsHtml}
                </div>
            </td>
        </tr>`;
    }).join('');
}

async function loadRecommendations(clusterId) {
    const cell = document.getElementById(`rec-cell-${clusterId}`);
    cell.innerHTML = '<span class="spinner-small" style="display:inline-block;"></span> 加载中...';

    try {
        const response = await fetch(`/api/recommend/${clusterId}`);
        const result = await response.json();

        if (result.code === 0 && result.data) {
            // Update cluster in state with recommendations
            const cluster = state.clusters.find(c => c.cluster_id === clusterId);
            if (cluster) {
                cluster.recommendations = result.data.recommendations;
            }

            const recommendations = result.data.recommendations || [];
            if (recommendations.length > 0) {
                cell.innerHTML = recommendations.slice(0, 3).map(rec => {
                    return `<div class="recommend-item">
                        <span class="recommend-course">${rec.course_name}</span>
                        <span class="recommend-score">${rec.score || 0}%</span>
                        <div class="recommend-reason">${rec.reason || ''}</div>
                    </div>`;
                }).join('') + `<button class="btn btn-ghost" style="margin-top:8px;font-size:0.75rem;" onclick="selectClusterAndChat(${clusterId})">咨询此群体 →</button>`;
            } else {
                cell.innerHTML = '<div style="color:var(--gray-400);">暂无推荐</div>';
            }
        } else {
            cell.innerHTML = `<div style="color:var(--danger);font-size:0.8rem;">${result.message || '加载失败'}</div>`;
        }
    } catch (err) {
        cell.innerHTML = `<div style="color:var(--danger);font-size:0.8rem;">加载失败</div>`;
    }
}

function selectClusterAndChat(clusterId) {
    selectClusterType(clusterId);
    goToChat();
}

function renderTypeChips(clusters) {
    console.log('[renderTypeChips] 开始渲染, clusters数量:', clusters.length);
    const container = document.getElementById('type-chips');
    container.innerHTML = clusters.map(cluster => {
        const isActive = state.currentCluster === cluster.cluster_id ? 'active' : '';
        return `<span class="type-chip ${isActive}" data-id="${cluster.cluster_id}" onclick="selectClusterType(${cluster.cluster_id})">
            ${cluster.label}
        </span>`;
    }).join('');
    console.log('[renderTypeChips] 渲染完成');
}

function selectClusterType(clusterId) {
    console.log('[selectClusterType] 开始, clusterId:', clusterId);
    state.currentCluster = clusterId;
    console.log('[selectClusterType] 1. state更新完成');

    // Update type chips
    document.querySelectorAll('.type-chip').forEach(chip => {
        chip.classList.toggle('active', parseInt(chip.dataset.id) === clusterId);
    });
    console.log('[selectClusterType] 2. type chips更新完成');

    // Update AI cards selection
    document.querySelectorAll('.ai-card').forEach(card => {
        card.classList.toggle('selected', parseInt(card.dataset.clusterId) === clusterId);
    });

    const cluster = state.clusters.find(c => c.cluster_id === clusterId);
    console.log('[selectClusterType] 3. 找到cluster:', cluster ? '是' : '否');
    if (cluster) {
        // Clear chat and add system message about selected type
        document.getElementById('chat-messages').innerHTML = `
            <div class="chat-message bot">
                <div class="message-avatar">🤖</div>
                <div class="message-content">已选择群体：${cluster.label}（${cluster.cluster_type}）。${cluster.description}</div>
            </div>
        `;
        console.log('[selectClusterType] 4. chat-messages更新完成');
    }

    // Enable chat input
    document.getElementById('chat-input').disabled = false;
    document.getElementById('chat-send').disabled = false;
    console.log('[selectClusterType] 5. 完成');
}

// ========================================
// Navigation
// ========================================
function goToChat() {
    // Make sure we have a selected cluster
    if (!state.currentCluster && state.clusters.length > 0) {
        selectClusterType(state.clusters[0].cluster_id);
    }
    updateProgress(4);
    showSection('section-chat');
    document.getElementById('chat-input').focus();
}

function backToClustering() {
    console.log('[backToClustering] 被调用');
    // 重置检查状态，允许再次点击"查看AI分析"
    isCheckingAnalysis = false;
    if (analysisCheckTimer) {
        clearInterval(analysisCheckTimer);
        analysisCheckTimer = null;
    }

    // 直接显示聚类结果，不显示等待页面
    updateProgress(3);
    document.getElementById('clustering-loading').style.display = 'none';
    document.getElementById('clustering-results').style.display = 'flex';
    showSection('section-clustering');
}

function backToPreview() {
    updateProgress(2);
    showSection('section-preview');
}

// ========================================
// Chat Functions (Non-streaming for reliability)
// ========================================
function addChatMessage(role, content) {
    const container = document.getElementById('chat-messages');
    const avatar = role === 'user' ? '👤' : '🤖';
    const messageClass = role === 'user' ? 'user' : 'bot';

    const messageEl = document.createElement('div');
    messageEl.className = `chat-message ${messageClass}`;
    messageEl.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">${escapeHtml(content)}</div>
    `;
    container.appendChild(messageEl);
    container.scrollTop = container.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function sendChat() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send');
    const message = input.value.trim();

    if (!message) return;
    if (state.currentCluster === null || state.currentCluster === undefined) {
        showToast('请先选择一个群体类型', 'error');
        return;
    }

    console.log('[发送聊天] cluster_id:', state.currentCluster, 'message:', message);

    // Disable input
    input.value = '';
    input.disabled = true;
    sendBtn.disabled = true;
    sendBtn.querySelector('.btn-text').style.display = 'none';
    sendBtn.querySelector('.btn-loading').style.display = 'inline-flex';

    // Add user message
    addChatMessage('user', message);

    // Add loading indicator for bot
    const botMessageEl = document.createElement('div');
    botMessageEl.className = 'chat-message bot';
    botMessageEl.id = 'bot-loading';
    botMessageEl.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content"><span class="spinner-small" style="display:inline-block;margin-right:8px;"></span>思考中...</div>
    `;
    document.getElementById('chat-messages').appendChild(botMessageEl);
    document.getElementById('chat-messages').scrollTop = document.getElementById('chat-messages').scrollHeight;

    try {
        const cluster = state.clusters.find(c => c.cluster_id === state.currentCluster);

        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                cluster_id: state.currentCluster,
                cluster_info: cluster,
                message: message
            })
        });

        // Remove loading indicator
        const loadingEl = document.getElementById('bot-loading');
        if (loadingEl) loadingEl.remove();

        if (response.ok) {
            const result = await response.json();

            if (result.code === 0 && result.data) {
                addChatMessage('bot', result.data.response);
            } else {
                addChatMessage('bot', '抱歉，' + (result.message || 'AI服务暂时不可用'));
            }
        } else {
            const result = await response.json();
            addChatMessage('bot', '抱歉，' + (result.message || 'AI服务暂时不可用'));
        }
    } catch (err) {
        // Remove loading indicator
        const loadingEl = document.getElementById('bot-loading');
        if (loadingEl) loadingEl.remove();

        addChatMessage('bot', '抱歉，网络出错：' + err.message);
        showToast('发送消息失败', 'error');
    } finally {
        input.disabled = false;
        sendBtn.disabled = false;
        sendBtn.querySelector('.btn-text').style.display = 'inline';
        sendBtn.querySelector('.btn-loading').style.display = 'none';
        input.focus();
    }
}

// ========================================
// Toast Notification
// ========================================
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = 'toast ' + type;

    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => toast.classList.remove('show'), 3000);
}