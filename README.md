# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution: 
> to students: please fill your solution description here.


### Chain design
receipt image
|
v
image_data_url()  (local file -> base64 data URL)
|
v
ChatPromptTemplate  (system: extract JSON {subtotal, discounts[], rounding, final_paid};
human: text + image_url)
|
v
ChatDeepSeek  (model="deepseek-v4-flash-vision-exp", temperature=0, max_tokens=8192)
|
v
JSON answer per receipt  ---- self-check: final_paid == subtotal + rounding ----
|                                                                 |
| pass (keep up to 5 samples, max 7 attempts)                     | fail -> retry
v
Aggregate per receipt:
paid     = median of the samples' final_paid
subtotal = median of the samples' subtotal
discounts = line-by-line median vote over the discount lists
(receipts print discount lines in a fixed order, so the i-th line
is the same discount in every sample; one misread line is outvoted)
|
v
Q1 = sum of paid per receipt              -> "HK$1974.30"
Q2 = sum of (subtotal + discounts)        -> "HK$2348.20"

### Description

The chain sends each receipt image to the vision-capable DeepSeek Flash model
once per sample and asks for a compact JSON object (subtotal, the list of
discount line amounts, rounding, and final paid amount) instead of free-form
text, so the two query answers can be computed with exact Decimal arithmetic
rather than trusting the model's own math. Because a vision model occasionally
misreads a digit or runs out of output budget on dense receipts, each receipt
is sampled up to 5 times; every sample must pass a self-check
(final_paid = subtotal + rounding) before it counts, and the per-receipt
figures are then combined by median voting - over the paid amounts, the
subtotals, and each discount line position separately - so a single bad
reading is outvoted by the majority. The two final answers are the sums of
these per-receipt medians, formatted as a single HK$ amount each.

