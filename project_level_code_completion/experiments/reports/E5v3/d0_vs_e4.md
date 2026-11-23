# D0 (Blackwell box) against E4 RAG F (RTX 5090) on the 24 dev datapoints

Lines: 1372, datapoints: 24. Writer input tokens identical on 100.0 % of lines.

| arm | EM datapoint mean | EM pooled |
|---|---:|---:|
| E4 RAG F (5090) | 0.5008 | 0.4257 |
| D0 RAG F (Blackwell) | 0.5037 | 0.4286 |

D0 minus E4: +0.29 points [+0.01, +0.69] (paired bootstrap over datapoints).
Predictions that differ: 40 of 1372 lines (2.9 %); EM differs on 6 (only E4 right 1, only D0 right 5).
