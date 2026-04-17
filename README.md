# Fitness Course Recommendation System

体测数据驱动的大语言模型体育课程个性化推荐系统

## 本地运行

```bash
pip install -r requirements.txt
python app.py
```

然后打开 http://localhost:5000

## 部署到 Hugging Face Spaces

1. 创建 GitHub 仓库并推送代码
2. 在 https://huggingface.co/new-space 创建 Space，选择 **Docker** 类型
3. 关联你的 GitHub 仓库
4. 在 Space 设置中添加环境变量 `MINIMAX_API_KEY`
