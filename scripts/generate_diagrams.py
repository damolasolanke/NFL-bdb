#!/usr/bin/env python3
"""Generate architecture diagrams for the NFL-bdb trajectory prediction project."""

import os

from diagrams import Cluster, Diagram, Edge
from diagrams.programming.language import Python
from diagrams.programming.framework import FastAPI
from diagrams.onprem.compute import Server
from diagrams.onprem.client import Client
from diagrams.generic.compute import Rack
from diagrams.generic.database import SQL

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "images")

CLUSTER_FONT = "Helvetica Neue Bold"
FONT = "Helvetica Neue"


def architecture_diagram():
    """STGNN model architecture — data to predictions."""
    with Diagram(
        "",
        filename=os.path.join(OUTPUT_DIR, "architecture"),
        outformat="png",
        show=False,
        direction="LR",
        graph_attr={
            "fontsize": "11",
            "fontname": FONT,
            "bgcolor": "#ffffff",
            "pad": "0.6",
            "nodesep": "0.5",
            "ranksep": "0.8",
            "dpi": "150",
        },
        node_attr={"fontsize": "10", "fontname": FONT},
        edge_attr={"fontsize": "8", "fontname": FONT, "color": "#666666"},
    ):
        # --- Data Preprocessing ---
        with Cluster(
            "Data  ·  22 players per frame",
            graph_attr={
                "bgcolor": "#E8F5E9",
                "style": "rounded",
                "fontsize": "11",
                "fontname": CLUSTER_FONT,
                "pencolor": "#2E7D32",
                "penwidth": "2.0",
            },
        ):
            raw = SQL("Tracking CSVs\nx, y, speed, dir, ...")
            features = Python("Feature Engineering\n13 node features")
            sequences = Python("Sequence Builder\n100-frame windows")

        # --- Graph Construction ---
        with Cluster(
            "Graph Construction  ·  PyTorch",
            graph_attr={
                "bgcolor": "#FFF3E0",
                "style": "rounded",
                "fontsize": "11",
                "fontname": CLUSTER_FONT,
                "pencolor": "#E65100",
                "penwidth": "2.0",
            },
        ):
            graph = Rack("Spatial Graph\n20-yard radius edges")
            edge_feat = Python("Edge Features\ndist · angle · rel_vel")

        # --- STGNN Model ---
        with Cluster(
            "STGNNRefine  ·  Model Architecture",
            graph_attr={
                "bgcolor": "#E3F2FD",
                "style": "rounded",
                "fontsize": "11",
                "fontname": CLUSTER_FONT,
                "pencolor": "#1565C0",
                "penwidth": "2.0",
            },
        ):
            encoder = Python("Node Encoder\nLinear → LayerNorm → GELU")
            with Cluster(
                "GAT Layers ×3  ·  8-head attention",
                graph_attr={
                    "bgcolor": "#BBDEFB",
                    "style": "rounded",
                    "fontsize": "10",
                    "fontname": CLUSTER_FONT,
                    "pencolor": "#1976D2",
                    "penwidth": "1.5",
                },
            ):
                gat = Rack("GATv2\nspatial interactions")
            with Cluster(
                "Temporal Transformer ×4",
                graph_attr={
                    "bgcolor": "#BBDEFB",
                    "style": "rounded",
                    "fontsize": "10",
                    "fontname": CLUSTER_FONT,
                    "pencolor": "#1976D2",
                    "penwidth": "1.5",
                },
            ):
                transformer = Rack("Pre-norm Encoder\npositional embeddings")
            refine = Python("Refinement Block\nFFN + residual")

        # --- Output ---
        with Cluster(
            "Inference  ·  Trajectory Output",
            graph_attr={
                "bgcolor": "#F3E5F5",
                "style": "rounded",
                "fontsize": "11",
                "fontname": CLUSTER_FONT,
                "pencolor": "#6A1B9A",
                "penwidth": "2.0",
            },
        ):
            head = Python("Output Head\n→ velocity Δx, Δy")
            positions = Python("Position Integration\ncumsum over time")
            submission = SQL("Kaggle Submission\nParquet output")

        # --- Edges: data flow ---
        raw >> Edge(color="#2E7D32", penwidth="2.0") >> features
        features >> Edge(color="#2E7D32", penwidth="2.0") >> sequences

        # --- Edges: graph ---
        sequences >> Edge(color="#E65100", penwidth="2.0") >> graph
        sequences >> Edge(color="#E65100", penwidth="1.5") >> edge_feat

        # --- Edges: model ---
        graph >> Edge(color="#1565C0", penwidth="2.0") >> encoder
        edge_feat >> Edge(color="#1565C0", style="dashed") >> gat
        encoder >> Edge(color="#1565C0", penwidth="2.0") >> gat
        gat >> Edge(color="#1565C0", penwidth="2.0") >> transformer
        transformer >> Edge(color="#1565C0", penwidth="2.0") >> refine

        # --- Edges: output ---
        refine >> Edge(color="#6A1B9A", penwidth="2.0") >> head
        head >> Edge(label="velocities", color="#6A1B9A", penwidth="2.0") >> positions
        positions >> Edge(label="positions", color="#6A1B9A", penwidth="1.5") >> submission


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating architecture diagram...")
    architecture_diagram()
    print(f"Done. Images in {OUTPUT_DIR}/")
