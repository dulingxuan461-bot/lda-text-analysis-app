# LDA 主题建模小程序

一个 Streamlit 文本分析小程序，用于快速完成 LDA 主题建模、词频词云、情感分析和词语共现网络分析。

## 功能

- 支持直接粘贴文本，或上传 `.txt` / `.csv` / `.xlsx` / `.xls`
- 支持自动识别、中文、英文、中英混合四种处理模式
- 中文使用 `jieba` 分词，英文按词切分并可选词干替换
- 可调主题数、词表大小、停用词、迭代次数等参数
- 可调 `α` / `β`、停用词、替换词、分词去重、词长过滤、英文词干替换
- 支持上传 `.txt` / `.csv` / `.xlsx` / `.xls` 停用词表和替换词表
- 支持 LDA 主题数量评估、主题权重词、主题气泡图、主题强度变化
- 支持总词频统计、词频下载、参考图蓝橙风格和多配色词云图
- 支持中英文文本情感分析和词语共现社会网络关系图
- 输出主题关键词、文档主题分布、主题占比图
- 可下载主题关键词和文档主题分布 CSV

## 启动

```bash
pip3 install -r requirements.txt
streamlit run app.py
```

启动后浏览器会打开本地页面。如果没有自动打开，可以访问终端里显示的地址。

## 让别人也能使用

如果希望别人不管用 Windows、macOS 还是 Linux 都能访问，推荐把它部署成网页。别人只需要浏览器，不需要安装 Python。

### 方式一：Streamlit Community Cloud

适合课程作业、论文演示和轻量共享。

1. 把整个项目上传到 GitHub 仓库。
2. 打开 Streamlit Community Cloud，新建 App。
3. 选择仓库和 `app.py`。
4. 部署后会得到一个公开网址，把网址发给别人即可。

项目里已经包含云端需要的文件：

- `requirements.txt`：Python 依赖
- `packages.txt`：Linux 中文字体依赖
- `runtime.txt`：Python 版本
- `.streamlit/config.toml`：Streamlit 配置

### 方式二：Docker 部署

适合部署到服务器、云主机、NAS 或任何支持 Docker 的电脑。

```bash
docker build -t lda-text-app .
docker run -p 8501:8501 lda-text-app
```

启动后访问：

```text
http://服务器IP:8501
```

如果部署在云服务器上，需要在安全组或防火墙里开放 `8501` 端口。

### 方式三：同一局域网临时共享

如果别人和你在同一个 Wi-Fi 下，可以这样启动：

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

然后把你电脑的局域网 IP 发给别人，例如：

```text
http://你的电脑IP:8501
```

这种方式只适合临时演示。电脑关机、断网或服务停止后，别人就无法访问。

## 输入格式

- `.txt`：每一行会被视为一篇文档；空行会被忽略
- `.csv` / `.xlsx` / `.xls`：上传后在界面中选择文本列
- 粘贴文本：每一行会被视为一篇文档

建议至少准备 20 篇以上文档；文档太少时，主题结果会更像关键词聚类，而不是真正稳定的主题结构。
