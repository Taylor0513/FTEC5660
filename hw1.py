#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    import os

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0,
        max_tokens=8192,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at reading supermarket receipts. "
                "From the receipt image, reply with a single JSON object and nothing "
                "else, in this exact shape:\n"
                '{{"subtotal": <number>, "discounts": [<number>, ...], "rounding": <number>, "final_paid": <number>}}\n'
                "- subtotal: the SUBTOTAL line value (after discounts, before rounding).\n"
                "- discounts: a list containing EVERY discount/promotion/coupon/savings "
                "line amount as a POSITIVE number, including multi-buy savings lines "
                "(like Buy 2 Save $5), percentage lines (like 5% OFF) and app/coupon "
                "discounts (like MB APP UPGRADE -$10). List each such line exactly once. "
                "Do NOT include the ROUNDING line here. Use [] if there are none.\n"
                "- rounding: the ROUNDING line value (usually a small negative number "
                "like -0.01). Use 0 if there is no ROUNDING line.\n"
                "- final_paid: the FINAL payment amount after ROUNDING (the amount next "
                "to the payment method such as OCTOPUS, VISA or CASH).\n"
                "Think briefly and efficiently. Check that no discount line is missed. "
                "Output JSON only.",
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "Read this supermarket receipt and extract the two amounts.",
                    },
                    {"type": "image_url", "image_url": {"url": "{image_data}"}},
                ],
            ),
        ]
    )

    return prompt | llm


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    inputs = [{"image_data": image_data_url(path)} for path in images]

    def extract(text, strict=True):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
            subtotal = Decimal(str(data["subtotal"]))
            discounts = [Decimal(str(item)) for item in data.get("discounts", [])]
            rounding = Decimal(str(data.get("rounding", 0)))
            paid = Decimal(str(data["final_paid"]))
            # self-check: paid should equal subtotal + rounding
            if strict and abs((subtotal + rounding) - paid) > Decimal("0.05"):
                return None
            return paid, subtotal, discounts
        except (KeyError, ValueError, InvalidOperation, TypeError):
            return None

    def median(values):
        ordered = sorted(values)
        n = len(ordered)
        return ordered[n // 2] if n % 2 == 1 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2

    def aggregate(parses):
        # per-receipt totals from several independent samples of the same receipt
        paid = median([p[0] for p in parses])
        subtotal = median([p[1] for p in parses])
        lengths = Counter(len(p[2]) for p in parses)
        best_len, _ = lengths.most_common(1)[0]
        same_len = [p[2] for p in parses if len(p[2]) == best_len]
        if len(same_len) >= 2:
            # vote line by line: receipts list discounts in a fixed order,
            # so the i-th discount is the same line in every sample
            discounts = sum(
                (median([d[i] for d in same_len]) for i in range(best_len)),
                Decimal("0"),
            )
        else:
            discounts = median([sum(p[2], Decimal("0")) for p in parses])
        return paid, subtotal + discounts

    total_paid = Decimal("0")
    total_no_discount = Decimal("0")

    for i, inp in enumerate(inputs):
        parses = []
        fallback = None
        attempts = 0
        while attempts < 7 and len(parses) < 5:
            attempts += 1
            out = chain.invoke(inp)
            text = response_text(out)
            parsed = extract(text)
            if parsed is not None:
                parses.append(parsed)
            else:
                loose = extract(text, strict=False)
                if loose is not None:
                    fallback = loose
        if parses:
            paid, without = aggregate(parses)
        elif fallback is not None:
            paid, without = fallback[0], fallback[1] + sum(fallback[2], Decimal("0"))
        else:
            continue
        total_paid += paid
        total_no_discount += without

    total_paid = total_paid.quantize(Decimal("0.01"))
    total_no_discount = total_no_discount.quantize(Decimal("0.01"))
    return {QUERY_1: f"HK${total_paid}", QUERY_2: f"HK${total_no_discount}"}


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
