# GG24 Deep Funding Level 1 - Graph-Augmented Blend Model

Gitcoin GG24 Deep Funding Level 1 模型提交  
融合方法：70% l1-predictions prior + 30% PageRank，歸一化 sum=1

## 模型概述
- 使用 networkx 計算反向 PageRank 捕捉依賴結構  
- 融合 l1-predictions.csv prior 權重  
- 最終輸出：submission_blended_70prior_30pr.csv（全98個repo覆蓋）

## 運行方式
### 依賴安裝
```bash
pip install pandas networkx scipy
