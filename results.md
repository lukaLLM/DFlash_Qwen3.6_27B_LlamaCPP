(LiveCodeBench) luke@luke-PC:~/Documents/Code/DFlash$ aiperf profile --model qwen36-dflash-ngram --url http://localhost:8001 \
  --endpoint-type chat --streaming --tokenizer Qwen/Qwen3.6-27B \
  --input-file benchmark/corpora/lcb_aug2024.jsonl --custom-dataset-type single_turn \
  --extra-inputs temperature:0 --concurrency 1 --random-seed 42 --request-count 30 \
  --output-artifact-dir artifacts/dflash/speed_lcb


                                                NVIDIA AIPerf | LLM Metrics                                                 
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃                                        Metric ┃      avg ┃    min ┃      max ┃      p99 ┃      p90 ┃      p50 ┃      std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│                      Time to First Token (ms) │   293.40 │ 201.52 │   551.95 │   503.04 │   373.77 │   274.67 │    68.06 │
│                     Time to Second Token (ms) │    32.84 │  31.64 │    35.48 │    35.22 │    34.30 │    32.60 │     0.94 │
│               Time to First Output Token (ms) │   293.40 │ 201.52 │   551.95 │   503.04 │   373.77 │   274.67 │    68.06 │
│                          Request Latency (ms) │ 3,459.57 │ 547.57 │ 8,673.50 │ 8,498.46 │ 7,710.40 │ 2,071.54 │ 2,818.52 │
│                      Inter Token Latency (ms) │     6.07 │   2.45 │    25.22 │    20.31 │     7.29 │     5.33 │     3.82 │
│              Output Token Throughput Per User │   195.18 │  39.64 │   408.99 │   391.11 │   269.54 │   187.51 │    69.84 │
│                             (tokens/sec/user) │          │        │          │          │          │          │          │
│ E2E Output Token Throughput (tokens/sec/user) │   156.13 │  21.16 │   243.84 │   231.91 │   193.13 │   154.90 │    36.62 │
│               Output Sequence Length (tokens) │   516.27 │  12.00 │ 1,024.00 │ 1,024.00 │ 1,024.00 │   329.00 │   391.07 │
│                Input Sequence Length (tokens) │   485.67 │ 262.00 │ 1,200.00 │ 1,059.35 │   695.90 │   435.00 │   179.65 │
│          Output Token Throughput (tokens/sec) │   148.98 │    N/A │      N/A │      N/A │      N/A │      N/A │      N/A │
│             Request Throughput (requests/sec) │     0.29 │    N/A │      N/A │      N/A │      N/A │      N/A │      N/A │
│                      Request Count (requests) │    30.00 │    N/A │      N/A │      N/A │      N/A │      N/A │      N/A │
└───────────────────────────────────────────────┴──────────┴────────┴──────────┴──────────┴──────────┴──────────┴──────────┘


No GPU telemetry data collected during the benchmarking run.

╭────────────────────────────── Output Sequence Length Mismatch Warning ──────────────────────────────╮
│  20 of 30 requests (66.7%) have output length differing from requested by more than the threshold.  │
│  Threshold (tokens): min(requested x 5%, 50)                                                        │
│  Average mismatch: -49.6%                                                                           │
│                                                                                                     │
│  Why: Server hit EOS token before reaching requested output length.                                 │
│                                                                                                     │
│  Fix Options:                                                                                       │
│    - --extra-inputs ignore_eos:true - Generate until max_tokens (vLLM, TensorRT-LLM)                │
│    - --extra-inputs min_tokens:<N> - Set minimum output length (vLLM, TensorRT-LLM, SGLang)         │
│    - --use-server-token-count - Use server-reported token counts if tokenizer mismatch suspected    │
│                                                                                                     │
│  Diagnostics:                                                                                       │
│    - Review profile_export.jsonl -> osl_mismatch_diff_pct for per-request values                    │
│    - Adjust: AIPERF_METRICS_OSL_MISMATCH_PCT_THRESHOLD=5                                            │
│    - Adjust: AIPERF_METRICS_OSL_MISMATCH_MAX_TOKEN_THRESHOLD=50                                     │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────╯

CLI Command: aiperf profile --model 'qwen36-dflash-ngram' --url 'http://localhost:8001' --endpoint-type 'chat' --streaming --tokenizer 'Qwen/Qwen3.6-27B' --input-file 
'benchmark/corpora/lcb_aug2024.jsonl' --custom-dataset-type 'single_turn' --extra-inputs 'temperature:0' --concurrency 1 --random-seed 42 --request-count 30 --output-artifact-dir 
'artifacts/dflash/speed_lcb'
Benchmark Duration: 103.96 sec
CSV Export: /home/luke/Documents/Code/DFlash/artifacts/dflash/speed_lcb/profile_export_aiperf.csv
JSON Export: /home/luke/Documents/Code/DFlash/artifacts/dflash/speed_lcb/profile_export_aiperf.json
Server Metrics CSV Export: /home/luke/Documents/Code/DFlash/artifacts/dflash/speed_lcb/server_metrics_export.csv
Server Metrics JSON Export: /home/luke/Documents/Code/DFlash/artifacts/dflash/speed_lcb/server_metrics_export.json
Log File: /home/luke/Documents/Code/DFlash/artifacts/dflash/speed_lcb/logs/aiperf.log

(LiveCodeBench) luke@luke-PC:~/Documents/Code/DFlash$ docker compose -f docker/docker-compose.yaml stop llamacpp_dflash
[+] stop 1/1
 ✔ Container LlamaCpp_Qwen36_DFlash Stopped                                                                                                                                                 0.7s
(LiveCodeBench) luke@luke-PC:~/Documents/Code/DFlash$ ./benchmark/speed_sweep.sh ngram          # full sweep: 512 / 4096 / 12288 / 36864
[+] up 1/1
 ✔ Container LlamaCpp_Qwen36_DFlash_Ngram Running                                                                                                                                           0.0s
>>> llamacpp_dflash_ngram  alias=qwen36-dflash-ngram


                                                NVIDIA AIPerf | LLM Metrics                                                 
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃                                        Metric ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p50 ┃    std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│                      Time to First Token (ms) │   357.12 │   335.92 │   390.31 │   386.27 │   365.04 │   358.36 │  10.59 │
│                     Time to Second Token (ms) │    33.98 │    28.48 │    94.46 │    76.76 │    33.14 │    32.40 │  11.32 │
│               Time to First Output Token (ms) │   357.12 │   335.92 │   390.31 │   386.27 │   365.04 │   358.36 │  10.59 │
│                          Request Latency (ms) │ 5,564.56 │ 2,701.96 │ 6,601.61 │ 6,558.23 │ 6,307.88 │ 5,682.15 │ 795.60 │
│                      Inter Token Latency (ms) │    10.19 │     4.62 │    12.22 │    12.13 │    11.64 │    10.43 │   1.55 │
│              Output Token Throughput Per User │   101.74 │    81.84 │   216.35 │   196.71 │   112.15 │    95.91 │  24.99 │
│                             (tokens/sec/user) │          │          │          │          │          │          │        │
│ E2E Output Token Throughput (tokens/sec/user) │    94.88 │    77.56 │   189.49 │   173.79 │   104.21 │    90.11 │  20.93 │
│               Output Sequence Length (tokens) │   512.00 │   512.00 │   512.00 │   512.00 │   512.00 │   512.00 │   0.00 │
│                Input Sequence Length (tokens) │   512.00 │   512.00 │   512.00 │   512.00 │   512.00 │   512.00 │   0.00 │
│          Output Token Throughput (tokens/sec) │    91.95 │      N/A │      N/A │      N/A │      N/A │      N/A │    N/A │
│             Request Throughput (requests/sec) │     0.18 │      N/A │      N/A │      N/A │      N/A │      N/A │    N/A │
│                      Request Count (requests) │    30.00 │      N/A │      N/A │      N/A │      N/A │      N/A │    N/A │
└───────────────────────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴────────┘


No GPU telemetry data collected during the benchmarking run.

CLI Command: aiperf profile --model 'qwen36-dflash-ngram' --url 'http://localhost:8001' --endpoint-type 'chat' --streaming --tokenizer 'Qwen/Qwen3.6-27B' --synthetic-input-tokens-mean 512 
--synthetic-input-tokens-stddev 0 --output-tokens-mean 512 --output-tokens-stddev 0 --extra-inputs 'temperature:0' --extra-inputs 'top_p:1.0' --extra-inputs 'top_k:1' --extra-inputs 
'ignore_eos:true' --extra-inputs 'min_tokens:512' --concurrency 1 --random-seed 42 --warmup-request-count 2 --request-count 30 --output-artifact-dir 
'/home/luke/Documents/Code/DFlash/benchmark/../artifacts/dflash_ngram/speed/isl512_osl512'
Benchmark Duration: 167.04 sec
CSV Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl512_osl512/profile_export_aiperf.csv
JSON Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl512_osl512/profile_export_aiperf.json
Server Metrics CSV Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl512_osl512/server_metrics_export.csv
Server Metrics JSON Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl512_osl512/server_metrics_export.json
Log File: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl512_osl512/logs/aiperf.log



                                                    NVIDIA AIPerf | LLM Metrics                                                    
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃                                        Metric ┃       avg ┃      min ┃       max ┃       p99 ┃       p90 ┃       p50 ┃      std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│                      Time to First Token (ms) │  1,682.57 │ 1,568.89 │  1,820.43 │  1,815.89 │  1,775.11 │  1,666.35 │    68.42 │
│                     Time to Second Token (ms) │     31.08 │    29.39 │     33.93 │     33.87 │     33.37 │     30.35 │     1.64 │
│               Time to First Output Token (ms) │  1,682.57 │ 1,568.89 │  1,820.43 │  1,815.89 │  1,775.11 │  1,666.35 │    68.42 │
│                          Request Latency (ms) │ 17,672.99 │ 9,639.19 │ 22,159.29 │ 22,042.58 │ 20,992.15 │ 18,426.57 │ 3,610.46 │
│                      Inter Token Latency (ms) │      3.91 │     1.95 │      5.01 │      4.98 │      4.70 │      4.10 │     0.88 │
│              Output Token Throughput Per User │    274.89 │   199.69 │    513.57 │    496.94 │    347.35 │    244.15 │    88.40 │
│                             (tokens/sec/user) │           │          │           │           │           │           │          │
│ E2E Output Token Throughput (tokens/sec/user) │    244.75 │   184.71 │    424.93 │    412.66 │    302.21 │    222.38 │    67.52 │
│               Output Sequence Length (tokens) │  4,090.30 │ 4,048.00 │  4,096.00 │  4,096.00 │  4,096.00 │  4,096.00 │    14.17 │
│                Input Sequence Length (tokens) │  4,096.00 │ 4,096.00 │  4,096.00 │  4,096.00 │  4,096.00 │  4,096.00 │     0.00 │
│          Output Token Throughput (tokens/sec) │    231.20 │      N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
│             Request Throughput (requests/sec) │      0.06 │      N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
│                      Request Count (requests) │     10.00 │      N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
└───────────────────────────────────────────────┴───────────┴──────────┴───────────┴───────────┴───────────┴───────────┴──────────┘


No GPU telemetry data collected during the benchmarking run.

CLI Command: aiperf profile --model 'qwen36-dflash-ngram' --url 'http://localhost:8001' --endpoint-type 'chat' --streaming --tokenizer 'Qwen/Qwen3.6-27B' --synthetic-input-tokens-mean 4096 
--synthetic-input-tokens-stddev 0 --output-tokens-mean 4096 --output-tokens-stddev 0 --extra-inputs 'temperature:0' --extra-inputs 'top_p:1.0' --extra-inputs 'top_k:1' --extra-inputs 
'ignore_eos:true' --extra-inputs 'min_tokens:4096' --concurrency 1 --random-seed 42 --warmup-request-count 2 --request-count 10 --output-artifact-dir 
'/home/luke/Documents/Code/DFlash/benchmark/../artifacts/dflash_ngram/speed/isl4096_osl4096'
Benchmark Duration: 176.92 sec
CSV Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl4096_osl4096/profile_export_aiperf.csv
JSON Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl4096_osl4096/profile_export_aiperf.json
Server Metrics CSV Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl4096_osl4096/server_metrics_export.csv
Server Metrics JSON Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl4096_osl4096/server_metrics_export.json
Log File: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl4096_osl4096/logs/aiperf.log



                                                    NVIDIA AIPerf | LLM Metrics                                                     
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃                                        Metric ┃       avg ┃       min ┃       max ┃       p99 ┃       p90 ┃       p50 ┃      std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│                      Time to First Token (ms) │  4,778.20 │  3,600.18 │  5,159.44 │  5,156.16 │  5,126.62 │  5,059.17 │   591.35 │
│                     Time to Second Token (ms) │     33.56 │     30.25 │     35.10 │     35.07 │     34.79 │     34.10 │     1.70 │
│               Time to First Output Token (ms) │  4,778.20 │  3,600.18 │  5,159.44 │  5,156.16 │  5,126.62 │  5,059.17 │   591.35 │
│                          Request Latency (ms) │ 37,419.03 │ 34,933.02 │ 40,638.17 │ 40,561.97 │ 39,876.15 │ 37,255.13 │ 2,090.16 │
│                      Inter Token Latency (ms) │      2.66 │      2.44 │      2.89 │      2.88 │      2.83 │      2.74 │     0.17 │
│              Output Token Throughput Per User │    378.02 │    346.32 │    410.41 │    410.13 │    407.61 │    365.09 │    24.65 │
│                             (tokens/sec/user) │           │           │           │           │           │           │          │
│ E2E Output Token Throughput (tokens/sec/user) │    329.40 │    302.38 │    351.76 │    351.52 │    349.37 │    329.83 │    18.15 │
│               Output Sequence Length (tokens) │ 12,288.00 │ 12,288.00 │ 12,288.00 │ 12,288.00 │ 12,288.00 │ 12,288.00 │     0.00 │
│                Input Sequence Length (tokens) │ 12,288.00 │ 12,288.00 │ 12,288.00 │ 12,288.00 │ 12,288.00 │ 12,288.00 │     0.00 │
│          Output Token Throughput (tokens/sec) │    327.90 │       N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
│             Request Throughput (requests/sec) │      0.03 │       N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
│                      Request Count (requests) │      5.00 │       N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
└───────────────────────────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴──────────┘


No GPU telemetry data collected during the benchmarking run.

CLI Command: aiperf profile --model 'qwen36-dflash-ngram' --url 'http://localhost:8001' --endpoint-type 'chat' --streaming --tokenizer 'Qwen/Qwen3.6-27B' --synthetic-input-tokens-mean 12288 
--synthetic-input-tokens-stddev 0 --output-tokens-mean 12288 --output-tokens-stddev 0 --extra-inputs 'temperature:0' --extra-inputs 'top_p:1.0' --extra-inputs 'top_k:1' --extra-inputs 
'ignore_eos:true' --extra-inputs 'min_tokens:12288' --concurrency 1 --random-seed 42 --warmup-request-count 1 --request-count 5 --output-artifact-dir 
'/home/luke/Documents/Code/DFlash/benchmark/../artifacts/dflash_ngram/speed/isl12288_osl12288'
Benchmark Duration: 187.38 sec
CSV Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl12288_osl12288/profile_export_aiperf.csv
JSON Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl12288_osl12288/profile_export_aiperf.json
Server Metrics CSV Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl12288_osl12288/server_metrics_export.csv
Server Metrics JSON Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl12288_osl12288/server_metrics_export.json
Log File: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl12288_osl12288/logs/aiperf.log



                                                    NVIDIA AIPerf | LLM Metrics                                                     
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃                                        Metric ┃       avg ┃       min ┃       max ┃       p99 ┃       p90 ┃       p50 ┃      std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│                      Time to First Token (ms) │ 15,504.26 │ 15,306.22 │ 15,792.86 │ 15,785.27 │ 15,717.03 │ 15,413.70 │   208.73 │
│                     Time to Second Token (ms) │     35.48 │     35.17 │     35.89 │     35.88 │     35.79 │     35.38 │     0.30 │
│               Time to First Output Token (ms) │ 15,504.26 │ 15,306.22 │ 15,792.86 │ 15,785.27 │ 15,717.03 │ 15,413.70 │   208.73 │
│                          Request Latency (ms) │ 32,384.45 │ 27,029.61 │ 35,838.08 │ 35,807.04 │ 35,527.60 │ 34,285.67 │ 3,839.12 │
│                      Inter Token Latency (ms) │      4.14 │      2.88 │      5.01 │      5.00 │      4.91 │      4.54 │     0.91 │
│              Output Token Throughput Per User │    255.78 │    199.71 │    347.17 │    344.64 │    321.83 │    220.46 │    65.17 │
│                             (tokens/sec/user) │           │           │           │           │           │           │          │
│ E2E Output Token Throughput (tokens/sec/user) │    127.80 │    113.85 │    150.61 │    149.98 │    144.28 │    118.94 │    16.26 │
│               Output Sequence Length (tokens) │  4,076.33 │  4,071.00 │  4,080.00 │  4,079.96 │  4,079.60 │  4,078.00 │     3.86 │
│                Input Sequence Length (tokens) │ 36,864.00 │ 36,864.00 │ 36,864.00 │ 36,864.00 │ 36,864.00 │ 36,864.00 │     0.00 │
│          Output Token Throughput (tokens/sec) │    125.81 │       N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
│             Request Throughput (requests/sec) │      0.03 │       N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
│                      Request Count (requests) │      3.00 │       N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
└───────────────────────────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴──────────┘


No GPU telemetry data collected during the benchmarking run.

╭───────────────────────────── Output Sequence Length Mismatch Warning ──────────────────────────────╮
│  3 of 3 requests (100.0%) have output length differing from requested by more than the threshold.  │
│  Threshold (tokens): min(requested x 5%, 50)                                                       │
│  Average mismatch: -88.9%                                                                          │
│                                                                                                    │
│  Why: Server hit EOS token before reaching requested output length.                                │
│                                                                                                    │
│  Fix Options:                                                                                      │
│    - --extra-inputs ignore_eos:true - Generate until max_tokens (vLLM, TensorRT-LLM)               │
│    - --extra-inputs min_tokens:<N> - Set minimum output length (vLLM, TensorRT-LLM, SGLang)        │
│    - --use-server-token-count - Use server-reported token counts if tokenizer mismatch suspected   │
│                                                                                                    │
│  Diagnostics:                                                                                      │
│    - Review profile_export.jsonl -> osl_mismatch_diff_pct for per-request values                   │
│    - Adjust: AIPERF_METRICS_OSL_MISMATCH_PCT_THRESHOLD=5                                           │
│    - Adjust: AIPERF_METRICS_OSL_MISMATCH_MAX_TOKEN_THRESHOLD=50                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────╯

CLI Command: aiperf profile --model 'qwen36-dflash-ngram' --url 'http://localhost:8001' --endpoint-type 'chat' --streaming --tokenizer 'Qwen/Qwen3.6-27B' --synthetic-input-tokens-mean 36864 
--synthetic-input-tokens-stddev 0 --output-tokens-mean 36864 --output-tokens-stddev 0 --extra-inputs 'temperature:0' --extra-inputs 'top_p:1.0' --extra-inputs 'top_k:1' --extra-inputs 
'ignore_eos:true' --extra-inputs 'min_tokens:36864' --concurrency 1 --random-seed 42 --warmup-request-count 1 --request-count 3 --output-artifact-dir 
'/home/luke/Documents/Code/DFlash/benchmark/../artifacts/dflash_ngram/speed/isl36864_osl36864'
Benchmark Duration: 97.20 sec
CSV Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl36864_osl36864/profile_export_aiperf.csv
JSON Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl36864_osl36864/profile_export_aiperf.json
Server Metrics CSV Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl36864_osl36864/server_metrics_export.csv
Server Metrics JSON Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl36864_osl36864/server_metrics_export.json
Log File: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/speed/isl36864_osl36864/logs/aiperf.log


(LiveCodeBench) luke@luke-PC:~/Documents/Code/DFlash$ ./benchmark/lcb_speed.sh ngram 100
[+] up 1/1
 ✔ Container LlamaCpp_Qwen36_DFlash_Ngram Started                                                                                                                                                                       0.1s
>>> llamacpp_dflash_ngram  alias=qwen36-dflash-ngram  warmup=1  measured=100/100
Failed to parse JSON string: '{' - JSONDecodeError('unexpected end of data: line 1 column 2 (char 1)')


                                                     NVIDIA AIPerf | LLM Metrics                                                     
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃                                        Metric ┃       avg ┃      min ┃        max ┃       p99 ┃       p90 ┃       p50 ┃       std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│                      Time to First Token (ms) │  1,029.31 │   236.33 │   1,305.11 │  1,301.27 │  1,228.95 │  1,093.63 │    222.71 │
│                     Time to Second Token (ms) │     69.13 │    28.11 │     168.88 │    116.00 │     97.57 │     83.20 │     30.34 │
│               Time to First Output Token (ms) │  1,029.31 │   236.33 │   1,305.11 │  1,301.27 │  1,228.95 │  1,093.63 │    222.71 │
│                          Request Latency (ms) │ 18,245.95 │ 1,125.63 │ 112,705.41 │ 70,390.25 │ 45,482.49 │ 12,205.03 │ 19,581.06 │
│                      Inter Token Latency (ms) │      6.21 │     2.64 │       8.46 │      8.28 │      7.60 │      6.49 │      1.28 │
│              Output Token Throughput Per User │    170.33 │   118.22 │     378.39 │    319.00 │    233.80 │    154.12 │     47.22 │
│                             (tokens/sec/user) │           │          │            │           │           │           │           │
│ E2E Output Token Throughput (tokens/sec/user) │    129.54 │    33.76 │     210.39 │    199.00 │    159.11 │    132.77 │     30.34 │
│               Output Sequence Length (tokens) │  2,511.86 │    38.00 │  16,200.00 │  9,331.38 │  6,544.00 │  1,620.50 │  2,732.99 │
│          Output Token Throughput (tokens/sec) │    137.58 │      N/A │        N/A │       N/A │       N/A │       N/A │       N/A │
│             Request Throughput (requests/sec) │      0.05 │      N/A │        N/A │       N/A │       N/A │       N/A │       N/A │
│                      Request Count (requests) │    100.00 │      N/A │        N/A │       N/A │       N/A │       N/A │       N/A │
└───────────────────────────────────────────────┴───────────┴──────────┴────────────┴───────────┴───────────┴───────────┴───────────┘


No GPU telemetry data collected during the benchmarking run.

CLI Command: aiperf profile --model 'qwen36-dflash-ngram' --url 'http://localhost:8001' --endpoint-type 'chat' --streaming --tokenizer 'Qwen/Qwen3.6-27B' --custom-dataset-type 'inputs_json' --input-file 
'/home/luke/Documents/Code/DFlash/benchmark/../artifacts/dflash_ngram/lcb_speed/inputs.patched.json' --dataset-sampling-strategy 'sequential' --concurrency 1 --random-seed 42 --warmup-request-count 1 --request-count 100 
--output-artifact-dir '/home/luke/Documents/Code/DFlash/benchmark/../artifacts/dflash_ngram/lcb_speed'
Benchmark Duration: 1825.77 sec
CSV Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/lcb_speed/profile_export_aiperf.csv
JSON Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/lcb_speed/profile_export_aiperf.json
Server Metrics CSV Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/lcb_speed/server_metrics_export.csv
Server Metrics JSON Export: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/lcb_speed/server_metrics_export.json
Log File: /home/luke/Documents/Code/DFlash/artifacts/dflash_ngram/lcb_speed/logs/aiperf.log

lcb q0 draft 248/589 accepted (42%), 260.2 tok/s
done -> /home/luke/Documents/Code/DFlash/benchmark/../artifacts/dflash_ngram/lcb_speed