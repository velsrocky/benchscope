# Quality test: Spark-X2.5-4B quants on a tough multi-skill prompt

Speed numbers don't show quality gaps, so all three quants faced the same
adversarial prompt. Method: `llama-cli` (build b10948-5f436dddb),
`-c 8192 -n 4096 -s 42 --temp 0` (greedy, fixed seed — fully reproducible),
same prompt file, one turn each.

## The prompt

> You are a precise engineering assistant. Follow every instruction exactly.
>
> TASK 1 - ARITHMETIC TRACE (show each multiplication step): A data center
> has 14 racks. Each rack holds 6 servers. Each server has 2 GPUs. Each GPU
> draws 312W at full load, plus each server has 180W of non-GPU overhead.
> Electricity costs $0.14 per kWh. What is the total hourly electricity cost
> of running everything at full load? Show each step, then put the final
> answer on its own line exactly as: ANSWER: $<value> (rounded to 2 decimals).
>
> TASK 2 - STRUCTURED TRANSFORM (output ONLY the JSON, no other text):
> Convert these server records to JSON sorted by uptime_days descending,
> keeping only hostname and uptime_days, and add rank starting at 1:
> web-03|41, db-01|112, cache-02|87
> Schema: {"servers":[{"rank":1,"hostname":"...","uptime_days":123}]}
> Rules: valid JSON, no trailing commas, no commentary.
>
> TASK 3 - CODE (output ONLY a python code block, nothing else): Write a
> Python function `median(xs)` that returns the median of a non-empty list
> of numbers WITHOUT using statistics, numpy, or sorted(); it must raise
> ValueError("empty") on empty input. On the last line, outside the
> function, print the result of median([7, 2, 9, 4]).

Ground truth: Task 1 = $9.46 (168 GPUs x 312W = 52416W; 84 servers x 180W =
15120W; 67536W = 67.536kW x $0.14 = $9.45504). Task 2 = db-01/112 rank 1,
cache-02/87 rank 2, web-03/41 rank 3. Task 3 = prints 5.5, raises on empty.

## Scorecard (RX 6800M, ROCm)

|  | BF16 (7.66 GiB) | Q8_0 (4.07 GiB) | Q4_K_M (2.42 GiB) |
|---|---|---|---|
| T1 math ($9.46) | PASS | PASS* | PASS |
| T2 exact JSON | FAIL (echoed schema only) | PASS | PASS |
| T3 code runs (5.5 + ValueError) | PASS | PASS (own bubble sort) | PASS (via `.sort()`) |
| Format discipline | partial | partial (closest to clean) | partial |
| Generation speed | 36.0 t/s | 58.8 t/s | 81.6 t/s |

## Observations

- All three get the arithmetic right with full step traces.
- The contradictory "output ONLY X" constraints (three of them, one
  response) send every quant into long meta-reasoning; **none finishes in
  1024 tokens** — budget 4096 for this kind of test.
- BF16, the "best" quant, is the only one that never emits the Task 2 JSON,
  despite 3x the VRAM and ~2.3x slower generation than Q4_K_M.
- Q8_0 lands closest to clean final artifacts (*math value right, but never on its own line – "So final answer: ANSWER: $9.46"); Q4_K_M's code leans on the
  `.sort()` method (within the letter of the ban, against its spirit).
- Bottom line for real-world use: no quality advantage for BF16 here.
  Q4_K_M matches it task-for-task at 2.3x the generation speed, ~1/3 the
  VRAM, and 2.5x the GPU-fit context (254K vs 101K).

Reproduce: save the prompt above to a file and run
`llama-cli -m <quant>.gguf -f prompt.txt -n 4096 -c 8192 -s 42 --temp 0 -st`.
