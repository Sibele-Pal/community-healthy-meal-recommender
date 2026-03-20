# src/run_pipeline.py

from src.data.ingest import run as run_ingest
from src.data.preprocess import run as run_preprocess
from src.graph.build_bipartite import run as run_graph
from src.data.join_region import run as run_join_region  # no-op but keeps structure


def main():
    print("=== PIPELINE START ===")
    run_ingest()
    run_preprocess()
    # join_region is now a stub but we call it to keep the chain visible
    run_join_region()
    run_graph()
    print("=== PIPELINE END ===")


if __name__ == "__main__":
    main()
