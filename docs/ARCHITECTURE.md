# 项目架构图

## 1. ETL 数据管线架构 (Medallion Architecture)

```mermaid
flowchart TD
    subgraph Bronze["🏗️ Bronze 层 — 原始数据摄取"]
        B1["📊 SEC EDGAR\nCompany Facts JSON"]
        B2["📈 Yahoo Finance\nDaily Prices CSV"]
        B3["💰 Yahoo Finance\nFinancial Statements CSV"]
        B4["🌐 OECD\nCbCR Raw CSV (466MB)"]
        B5["📚 Literature PDFs\n四大 + 学术论文 ×11"]
    end

    subgraph Silver["🔧 Silver 层 — 清洗 & 星型模式"]
        S_DIM["📋 维度表\n──────\ndim_firm\ndim_date\ndim_jurisdiction\ndim_policy_event"]
        S_FACT["📊 事实表\n──────\nfirm_financial_year\nmarket_daily + monthly\nfirm_revenue_year\ncbcr_jurisdiction_year"]
    end

    subgraph Gold["⭐ Gold 层 — 分析级数据 & 产出"]
        G_ANALYTICAL["🔬 Analytical\n──────\nexposure_score\nevent_firm_car\ndim_model\ndim_analysis_firm"]
        G_STATS["📈 Statistical\n──────\nPython DiD/SCM/SDiD\nR 交错DiD/gsynth/grf\nverified_model_results"]
        G_FIGURES["📊 Figures & Scoreboards\n──────\n21张出版物图表 (PNG+PDF)\nscoreboard.csv + md"]
    end

    subgraph Serving["🚀 Serving 层 — 消费"]
        SV_AI["🤖 AI Serving\n──────\nrag_corpus.csv\ncitation_registry\nvector_store\nreport.md"]
    end

    Bronze --> Silver
    Silver --> Gold
    Gold --> Serving

    style Bronze fill:#D6E4F0,stroke:#2F5597
    style Silver fill:#E2EFDA,stroke:#548235
    style Gold fill:#FFF2CC,stroke:#BF9000
    style Serving fill:#F2DCDB,stroke:#C00000
```

## 2. ETL + AI 集成架构

```mermaid
flowchart LR
    subgraph DataPipeline["📦 数据管线 (ETL)"]
        direction TB
        BR["Bronze\n原始数据"]
        SL["Silver\n清洗后数据"]
        GL["Gold\n分析产出"]
        BR --> SL --> GL
    end

    subgraph Analysis["🔬 分析引擎"]
        direction TB
        PY["Python\nDiD · SCM · Event Study\nExposure Design · Weighted DiD"]
        RV["R Validation\nCS DiD · SA DiD · gsynth\nHonestDiD · GRF · SDiD\nBacon Decomposition"]
        VER["Verification\nPython/R Cross-check\nVerified Results JSON"]
    end

    subgraph RAG["🧠 RAG 知识库"]
        direction TB
        CORPUS["语料构建\nPDF按页索引 + 文献元数据\n模型结果JSON + CSV\n图表元数据"]
        EMBED["向量化\nTF-IDF + 语义嵌入\n(sentence-transformers)"]
        RETRIEVE["检索\n混合评分 · 文献过滤\nCosine + Embedding"]
        CORPUS --> EMBED --> RETRIEVE
    end

    subgraph Report["📝 报告生成 (防幻视)"]
        direction TB
        EVIDENCE["证据链溯源\n声明 → Citation → 模型结果 → 数据源"]
        CITATION["引用验证\nID注册表校验 · 文件哈希比对"]
        OUTPUT["最终报告\n决策支持报告 + 证据溯源附录"]
    end

    DataPipeline -->|"Silver/Gold 数据"| Analysis
    Analysis -->|"verified_results.json"| RAG
    DataPipeline -->|"文献PDF + 模型产出"| RAG
    RAG -->|"检索文献 + 模型证据"| Report
    Analysis -->|"统计结果"| Report
    Report -->|"防幻视校验"| OUTPUT

    style DataPipeline fill:#D6E4F0,stroke:#2F5597
    style Analysis fill:#E2EFDA,stroke:#548235
    style RAG fill:#FFF2CC,stroke:#BF9000
    style Report fill:#F2DCDB,stroke:#C00000
    style OUTPUT fill:#E4DFEC,stroke:#7030A0
```

## 图例说明

| 颜色 | 含义 | 对应架构层 |
|:---|:---|:---|
| 🔵 蓝色 | 数据存储 / ETL | Bronze · Silver · Gold |
| 🟢 绿色 | 分析计算 | Python · R · Verification |
| 🟡 黄色 | 知识检索 | RAG · Embedding · Retrieval |
| 🔴 红色 | 报告生成 | Evidence Chain · Citation · Report |
| 🟣 紫色 | 最终交付 | Decision Support Report |

## SVG 文件

高分辨率 SVG 版本已生成在：
- `data/gold/figures/Figure_arch_etl_pipeline.svg`
- `data/gold/figures/Figure_arch_etl_ai_integration.svg`
