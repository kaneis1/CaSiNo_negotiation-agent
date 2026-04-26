# Day 9.2 Big Five Distribution Gate

Role: `mturk_agent_1`
Train file: `data/casino_train_w0.2.json`
Test file: `data/casino_test.json`

## Legacy Rule
Train counts: `{'cooperative': 357, 'competitive': 38, 'balanced': 121}`
Test counts: `{'cooperative': 115, 'competitive': 14, 'balanced': 21}`

## Train-Locked Active Rule
Train counts: `{'cooperative': 163, 'competitive': 177, 'balanced': 176}`
Test counts: `{'cooperative': 57, 'competitive': 38, 'balanced': 55}`
Gate passed: `True`

The active thresholds are selected on train metadata only and then frozen before reporting the test distribution.
