# Hand-check of AP@5 (fusion, heavy viewer)

user 10008352999832730058
relevant (actual top-5): [12001682, 12002856, 12001734, 12002355, 12002771]
recommended: [12001682, 12002955, 12002856, 12002896, 2400480]

| rank | id | relevant? | hits so far | precision at rank |
|---|---|---|---|---|
| 1 | 12001682 | yes | 1 | 1/1 = 1.000 |
| 2 | 12002955 | no | 1 | 1/2 = 0.500 |
| 3 | 12002856 | yes | 2 | 2/3 = 0.667 |
| 4 | 12002896 | no | 2 | 2/4 = 0.500 |
| 5 | 2400480 | no | 2 | 2/5 = 0.400 |

AP@5 = (1/1 + 2/3) / min(5, 5) = 0.3333  (value from common.apk)