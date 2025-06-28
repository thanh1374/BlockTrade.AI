import os
import json
import requests
from collections import defaultdict
from app.main import make_api_url, genLocation, ranker, trace_related_nodes, prepareGraph
from collections import deque

def visualize_wallet(wallet_address: str, depth: int) -> tuple[bool, str]:
    try:
        if not wallet_address:
            return False, "❌ Wallet address cannot be empty."
        if not isinstance(depth, int) or depth <= 0:
            return False, "❌ Depth must be a positive integer."

        center_node = wallet_address.strip().lower()
        max_depth = depth

        # Call Etherscan API to get ETH transactions
        url = make_api_url("account", "txlist", center_node, startblock=0, endblock=99999999, sort="asc")
        res = requests.get(url)
        data = res.json()

        if data["status"] != "1":
            return False, f"❌ API Error: {data.get('message', 'No data returned.')}"

        transactions = data["result"]

        # Build raw_links from transactions
        raw_links = defaultdict(dict)
        for tx in transactions:
            from_addr = tx.get("from", "").lower().strip()
            to_addr = tx.get("to", "").lower().strip()
            value_eth = int(tx.get("value", 0)) / 1e18

            if from_addr and to_addr and value_eth > 0:
                raw_links[from_addr][to_addr] = raw_links[from_addr].get(to_addr, 0) + value_eth

        # Filter top-N links per node
        top_links = ranker(raw_links, top=10)

        # Trace the subgraph from the center wallet
        traced_nodes, traced_edges = trace_related_nodes(top_links, center_node, max_depth)

        # Build Sigma.js compatible graph structure
        graph_json = {"nodes": [], "edges": []}
        edge_id = 0
        processed_edges = set()

        # Add nodes
        for node in traced_nodes:
            if not node:
                continue  # Skip empty node
            x, y = genLocation()
            graph_json["nodes"].append({
                "label": node,
                "x": x,
                "y": y,
                "id": f"id={node}",
                "size": 15 if node == center_node else 10
            })

        # Add edges
        for from_addr, to_addr, value in traced_edges:
            if not from_addr or not to_addr:
                continue  # Skip invalid edges
            key = f"{from_addr}:{to_addr}"
            if key in processed_edges:
                continue
            size = min(value, 20)
            graph_json["edges"].append({
                "source": f"id={from_addr}",
                "target": f"id={to_addr}",
                "id": edge_id,
                "size": size / 3 if size > 3 else size
            })
            edge_id += 1
            processed_edges.add(key)

        # Write to graph.json.js
        output_path = os.path.join("app", "libs", "graph.json.js")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        prepareGraph(output_path, json.dumps(graph_json))

        return True, f" Graph contains `{len(graph_json['nodes'])}` nodes."

    except Exception as e:
        return False, f"❌ Error: {str(e)}"
