# 文档分块策略对中文规章制度 RAG 问答效果的影响研究

梁颢严 · 2307054220 · B 类课程实验 · 2026-10

## 下载完整复现材料

仓库根目录提供完整复现 ZIP、报告 PDF 和可编辑 DOCX。ZIP 包含全部数据、模型、原始 API 响应、图表及哈希清单。运行下面的离线复核或重跑命令前，请先解压 ZIP 并进入 `rag-chunk-study` 目录。仓库中的 `src` 等文件便于在线阅读；完整运行以 ZIP 为准。报告与归档保存原实验版本，仓库于 2026-10-15 上传。

## 先读结果

报告 PDF / DOCX 与本材料包一起提供。数据是中山大学公开历史制度；本项目不提供当前政策咨询。
- `results/summary.json`：初版评估结果。
- `results/summary_v2.json`：修订评估结果；原始回答不变。
- `results/retrieval.json`：9 配置 × 2 协议 × 39 题，共 702 条检索记录。
- `results/generations`：390 次主实验 + 36 次重复，426 条真实响应。
- `results/judgments`：426 条初版评分。
- `results/rubrics_v2`、`judgments_v2`：只看原文生成的评分要点和修订评分。
- `EVALUATION_V2.md`：评分修订协议；不换题、不重新挑选参数、不重写答案。

模型评分不等于人工真值；尤其第一版金标准外断言字段不能解释为幻觉率。全文结论以报告的限定条件为准。

## 软件与数据

运行实验：Docker、Python 3.11、NumPy、ONNX Runtime、tokenizers、requests、PyMuPDF、BeautifulSoup；无需 GPU、向量数据库或训练框架。
`Dockerfile` 固定基础镜像摘要与关键依赖版本；构建需要联网下载镜像和依赖。实验本身在内部网络运行，仅中继访问固定 API 地址。
`models/provenance.json` 固定 ONNX 模型仓库、revision 和文件哈希。`data/source_manifest.json` 列出4份制度及官方问答的原始链接、日期与文件哈希。原始资料在 `data/raw`，清洗语料、候选问答与39题证据集在 `data/processed`。问答 PDF 不进入检索库。版权分别归原权利人，模型遵循其仓库许可；不得将第三方材料误称为本人原创。

## 一分钟离线复核（不需要密钥）

在解压目录运行，Python 3.11/3.12 均可：

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install numpy==2.2.6
python src/verify_offline.py
python src/analyze.py
python src/analyze.py --v2
```

前者检查分块覆盖、长度、坐标及源文件哈希；后者从原始响应重新计算统计表、配对 bootstrap 和重复实验结果。CSV 使用 UTF-8 BOM，便于表格软件打开。脚本会重写汇总文件，但不修改原始模型响应。

## 重新运行检索（不调用外部模型）

在 Linux Docker 主机运行：

```sh
docker build -t rag-chunk-study:20261010 .
mkdir -p rerun-results
sudo chown 10001:10001 rerun-results
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --memory 768m --cpus 1 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -v "$PWD/src:/workspace/src:ro" -v "$PWD/data:/workspace/data:ro" \
  -v "$PWD/models:/workspace/models:ro" -v "$PWD/rerun-results:/workspace/results:rw" \
  rag-chunk-study:20261010 python src/experiment.py retrieve
```

对比 `rerun-results/retrieval.json` 与随附结果。运行时间可变化，跨硬件浮点差异可能影响相似分数排序。重新清洗及生成冻结标注可用同一镜像挂载可写 `data`，依次执行 `src/prepare.py`、`src/annotate.py`；建议在副本操作以保留数据快照。

## 完整重跑（需要本人提供 API 密钥，会产生费用）

`start_gateway.sh` 与 `run.sh` 记录本次服务器运行参数，路径固定为 `/opt/rag-chunk-study-20261010`；复制到该目录或修改两脚本中的路径后使用。不要在有同名容器的主机直接重复创建。建议另用专用目录、容器名及新结果目录保存复现实验。

在专用目录创建 `secrets`，用交互式读取避免密钥进入命令历史（bash）：

```sh
mkdir -p secrets
chmod 700 secrets
read -rsp 'DeepSeek API key: ' RAG_EXPERIMENT_KEY
printf '%s' "$RAG_EXPERIMENT_KEY" > secrets/deepseek_key
unset RAG_EXPERIMENT_KEY
chmod 600 secrets/deepseek_key
sudo chown 10001:10001 secrets/deepseek_key
sudo sh start_gateway.sh
sudo sh run.sh
```

`run.sh` 会创建后台实验容器。使用 `docker logs -f rag-chunk-experiment` 查看状态，只有退出码 0 且426条输出和426条初版评分齐全才代表主实验完成。每个响应按题号缓存；更换模型、提示、语料或预算时必须使用新的结果目录，不能混用旧缓存。

修订评估：保持中继运行，在与实验相同的网络、资源限制和只读挂载下，用 `python src/evaluate_v2.py` 作为启动命令。它先生成39题评分要点并校验引文，再对426条已有答案评分。不要把修订分数与原始分数混为同一次评估。

结束后删除中继容器和临时密钥：

```sh
sudo docker stop rag-chunk-gateway
sudo docker rm rag-chunk-gateway
sudo rm secrets/deepseek_key
```

本次已执行清理；材料包不含密钥。接口别名为 `deepseek-chat`，记录返回 `deepseek-flash`。在线模型行为与接口映射可能变化；附原始响应可复核历史结果，不能保证未来重跑得到逐字相同答案。

## 评测范围

主比较预先固定为256字符，Top-3与严格768正文字符预算。固定窗口不重叠；重叠率20%；条款规则按标点拆长条、顺序合并短条。相同字符数不是相同 token 数。证据单元90%覆盖为召回阈值，可能遗漏重要否定词；应结合连续覆盖和原文案例。39题非随机大样本、同源模型评分、共享条款相关性均限制外推。

## 代码使用

本项目原创 Python 与 shell 代码以 MIT 许可提供（见 LICENSE）。第三方文档、模型和模型响应不自动继承该许可。本仓库当前为私有仓库，不应据此声称已满足公开开源仓库加分要求。

重新下载公开文档可运行 `python src/download_sources.py`，默认写入 `data/downloaded_raw`，不覆盖实验快照；脚本依赖 curl，并报告与原快照的哈希是否一致。
