import pandas as pd
import networkx as nx

print("=== GG24 Deep Funding Level 1 Baseline + Blend 模型開始運行 ===")

# === 檔案路徑（假設都在同資料夾） ===
seeds_path = 'repos_to_predict.csv'
edges_path = 'unweighted_graph.csv'
prior_path = 'l1-predictions.csv'   # 你的示例權重檔案

try:
    # 1. 載入 98 seeds
    seeds_df = pd.read_csv(seeds_path)
    seed_urls = seeds_df['repo'].tolist()
    seed_names = [url.replace('https://github.com/', '').rstrip('/') for url in seed_urls]
    seed_set = set(seed_names)

    print(f"成功載入 seeds：{len(seed_names)} 個")

    # 2. 載入 dependency graph edges
    edges_df = pd.read_csv(edges_path, encoding='utf-8')

    # 丟掉無用的 Unnamed: 0 欄位（如果存在）
    if 'Unnamed: 0' in edges_df.columns:
        edges_df = edges_df.drop(columns=['Unnamed: 0'])

    print("Edges 欄位名稱：", edges_df.columns.tolist())
    print("總 edges 行數：", len(edges_df))
    print("\n前 5 行樣本：")
    print(edges_df.head(5).to_string(index=False))

    # 建構 seed_full = owner/repo
    edges_df['seed_full'] = edges_df['seed_repo_owner'] + '/' + edges_df['seed_repo_name']

    # 過濾匹配你的 98 seeds
    relevant_edges = edges_df[edges_df['seed_full'].isin(seed_set)]

    print(f"\n匹配到的 edges 數：{len(relevant_edges)}")
    print(f"覆蓋的 seeds 數：{len(relevant_edges['seed_full'].unique())} / 98")

    if len(relevant_edges) == 0:
        raise ValueError("沒有任何匹配，請檢查 seeds 名稱格式或 graph 檔案")

    # 3. 建構有向圖：seed -> package (depends on)
    G = nx.DiGraph()

    for _, row in relevant_edges.iterrows():
        seed = row['seed_full']
        if pd.notna(row['package_repo_owner']) and pd.notna(row['package_repo_name']):
            pkg = f"{row['package_repo_owner']}/{row['package_repo_name']}"
        else:
            pkg = row['package_name']
        G.add_edge(seed, pkg)

    print(f"圖建構完成：節點 {G.number_of_nodes()}，邊 {G.number_of_edges()}")

    # 4. 反向圖 + PageRank（被依賴越多越重要）
    G_rev = G.reverse()
    pr = nx.pagerank(G_rev, alpha=0.85, max_iter=1000, tol=1e-8)

    # 5. 提取 seeds 的 PageRank 分數
    seed_pr = {}
    for url, name in zip(seed_urls, seed_names):
        score = pr.get(name, 0.0)
        seed_pr[url] = score

    # 6. 純 PageRank 歸一化
    total_pr = sum(seed_pr.values())
    pr_weights = {url: score / total_pr if total_pr > 0 else 1.0 / len(seed_pr) for url, score in seed_pr.items()}

    # 輸出純版 CSV
    pr_df = pd.DataFrame({
        'repo': list(pr_weights.keys()),
        'parent': 'ethereum',
        'weight': list(pr_weights.values())
    }).sort_values('weight', ascending=False)

    pr_df.to_csv('submission_pure_pagerank.csv', index=False, float_format='%.10f')
    print("\n純 PageRank 前 10 高權重：")
    print(pr_df.head(10)[['repo', 'weight']].to_string(index=False))
    print(f"純 PageRank 總和：{pr_df['weight'].sum():.10f}")

    # === 融合 l1-predictions.csv（推薦使用） ===
    prior_df = pd.read_csv(prior_path)
    prior_weights = dict(zip(prior_df['repo'], prior_df['weight']))

    blended = {}
    for url in seed_urls:
        pr_score = seed_pr.get(url, 0.0)
        prior_score = prior_weights.get(url, 0.0)
        blended[url] = 0.7 * prior_score + 0.3 * pr_score   # 70% prior + 30% pr

    total_blended = sum(blended.values())
    final_weights = {url: v / total_blended if total_blended > 0 else 1.0 / len(blended) for url, v in blended.items()}

    # 輸出融合版 CSV
    final_df = pd.DataFrame({
        'repo': list(final_weights.keys()),
        'parent': 'ethereum',
        'weight': list(final_weights.values())
    }).sort_values('weight', ascending=False)

    final_df.to_csv('submission_blended_70prior_30pr.csv', index=False, float_format='%.10f')

    print("\n=== 融合版結果（推薦提交這個） ===")
    print("融合前 10 高權重：")
    print(final_df.head(10)[['repo', 'weight']].to_string(index=False))
    print(f"融合總權重和：{final_df['weight'].sum():.10f}")
    print("已生成兩個檔案：")
    print("1. submission_pure_pagerank.csv （純 PageRank）")
    print("2. submission_blended_70prior_30pr.csv （融合版，覆蓋全 98 個）")

except Exception as e:
    print("\n運行錯誤：", str(e))
    print("請檢查：")
    print("1. 三個檔案是否都在同資料夾：repos_to_predict.csv, unweighted_graph.csv, l1-predictions.csv")
    print("2. pandas / networkx / scipy 已安裝（用 pip install）")
    print("3. 檔案內容是否完整（尤其是 l1-predictions.csv 有 98 行 weight）")