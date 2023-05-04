# Readme

# Dynamic Temporal Resolution for Video Retrieval: Incorporating Temporal Granularity from Fine-to-Coarse Levels.

The implementation of paper,  Dynamic Temporal Resolution for Video Retrieval: Incorporating Temporal Granularity from Fine-to-Coarse Levels.

## Introduction

We introduce a novel method, Temproa Resolution Adaptive Positional Encoder (TRAPE) to incorporate positional embedding with multiple different time scales, and Dynamic Temporal Resolution Encoder (DTRe) to effectively comprehend frame features from various temporal resolutions.

We implemented our method on three open-sourced works, [X-CLIP](https://github.com/xuguohai/X-CLIP), [TS2NET](https://github.com/yuqi657/ts2_net) and [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip).

## How to Run

1) data download

Please refer to the guidelines provided in CLIP4Clip

2) Checkpoint models

You can download the public pre-trained checkpoints for CLIP in the official homepage.

Additionally, we  are preparing to provide the trained checkpoint models for our  work.

3) Run scripts

### MSR-VTT

```jsx
sh run_msrvtt.sh

```

### LSMDC

```jsx
sh run_lsmdc.sh
```

### MSVD

```jsx
sh run_msvd.sh

```


## Experimental Results

### MSR-VTT

|  | R@1 | R@5 | R@10 | MeanR |
| --- | --- | --- | --- | --- |
| X-CLIP+DTRe | 48.1 | 74.4 | 83.6 | 13.3 |
| TS2NET+DTRe+DSL | 51.3 | 76.8 | 85.6 | 11.0 |
| CLIP4Clip | 43.9 | 71.9 | 80.8 | 14.6 |

### LSMDC

|  | R@1 | R@5 | R@10 | MeanR |
| --- | --- | --- | --- | --- |
| X-CLIP+DTRe | 24.5 | 42.8 | 51.3 | 57.2 |
| TS2NET+DTRe+DSL | 23.0 | 40.4 | 49.2 | 67.8 |
| CLIP4Clip | 23.4 | 41.3 | 49.3 | 60.1 |

### MSVD

|  | R@1 | R@5 | R@10 | MeanR |
| --- | --- | --- | --- | --- |
| X-CLIP+DTRe | 46.8 | 76.6 | 85.1 | 10.1 |
| CLIP4Clip | 46.5 | 75.9 | 84.7 | 10.0 |

## Acknowledgements

The implementation of our work relies on resources of CLIP, CLIP4Clip, TS2NET and X-CLIP. 

We thank the authors for their open-sourcing.
