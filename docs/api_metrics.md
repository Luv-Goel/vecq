# Metrics & Utilities

## vecq.metrics

vecq implements three core distance metrics. All metrics are designed to return a **distance** (where smaller values mean the vectors are closer together). This allows the internal priority queues to sort uniformly regardless of the metric chosen.

::: vecq.metrics
    options:
      show_root_heading: false
      show_source: true
      members:
        - cosine_distance
        - l2_distance
        - inner_product_distance
        - normalize

## vecq.bench

The benchmarking utilities used to generate the recall-vs-qps statistics.

::: vecq.bench
    options:
      show_root_heading: false
      show_source: false
      members:
        - run_benchmark
        - make_dataset
        - format_table
        - plot_recall_vs_qps
