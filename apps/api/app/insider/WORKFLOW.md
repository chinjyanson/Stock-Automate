# Insider Trading Agent — workflow contract

> **Renamed from `CLAUDE.md` when this folder was vendored into the trading platform.**
> A nested `CLAUDE.md` is auto-loaded as *instructions* by any Claude Code session
> working in this repository, and the text below is an executable workflow that
> posts to Slack and spends Firecrawl credits. Left under its original name it
> would read as a standing order rather than as reference material. It is
> documentation here; nothing in this repository runs it.

Instructions for the Claude Code routine that runs this pipeline. **This file is the
workflow contract** — the routine's saved prompt is *"Please execute the workflow in
the claude.md file,"* so everything below is what you (the routine) do on each fire.

## What triggers you

A **Firecrawl page monitor** watches <https://www.secform4.com/insider-sales> once an
hour and, after each check completes, POSTs to this routine's fire endpoint. **Firecrawl
fires you on every hourly check — changed or not.** It does **not** pass any input, so
you must decide for yourself whether a genuinely new insider-sale filing appeared. Most
fires (roughly 23 of every 24) will find nothing new: **exit fast in that case.** Do real
work only when there is a new row.

Dedup is free: Firecrawl only labels the page `changed` when its content actually differs
from the previous snapshot, so consecutive `same` checks never re-flag a row you already
processed. You do **not** need to keep your own last-seen state.

## Constants

- `MONITOR_ID` = `019f48e3-702a-733c-9b8d-8f42fd77da9e`
- Firecrawl API base = `https://api.firecrawl.dev`
- Detection log = `detections/insider-sales.jsonl` (in this repo)

## Prerequisite

`FIRECRAWL_API_KEY` must be set in this routine's environment (add it as a secret in the
routine's settings on claude.ai). If it is missing, **exit gracefully** — do not error out.

## Workflow

Run these steps in order. `jq` is used for JSON parsing.

### Step 0 — Preflight

```bash
if [ -z "$FIRECRAWL_API_KEY" ]; then
  echo "FIRECRAWL_API_KEY not set in routine environment — cannot query the monitor. Exiting."
  exit 0
fi
AUTH="Authorization: Bearer $FIRECRAWL_API_KEY"
MONITOR_ID=019f48e3-702a-733c-9b8d-8f42fd77da9e
BASE=https://api.firecrawl.dev
```

### Step 1 — Get the check that just completed

Fetch recent completed checks and pick the newest by `finishedAt`:

```bash
CHECKS=$(curl -s "$BASE/v2/monitor/$MONITOR_ID/checks?status=completed&limit=5" -H "$AUTH")
CHECK_ID=$(echo "$CHECKS" | jq -r '.data | sort_by(.finishedAt) | last | .id')
CHANGED=$(echo "$CHECKS" | jq -r '.data | sort_by(.finishedAt) | last | .summary.changed')
echo "Latest completed check: $CHECK_ID (changed pages: $CHANGED)"
```

### Step 2 — Gate on "did anything change at all?"

```bash
if [ "$CHANGED" = "0" ] || [ -z "$CHANGED" ] || [ "$CHANGED" = "null" ]; then
  echo "No changed pages in the latest check — no new filings. Exiting."
  exit 0
fi
```

### Step 3 — Confirm it's a *meaningful new row*, not page churn

Pull the changed page(s) and inspect the AI judgment. The monitor's goal instructs the
judge to flag **only** when a new insider-sale row is added, so require
`judgment.meaningful == true`. Treat it as a real detection when the judgment is meaningful
**and** the diff adds one or more table rows.

```bash
DETAIL=$(curl -s "$BASE/v2/monitor/$MONITOR_ID/checks/$CHECK_ID?status=changed" -H "$AUTH")

MEANINGFUL=$(echo "$DETAIL" | jq -r '[.data.pages[]? | select(.judgment.meaningful == true)] | length')
# Added markdown table rows in the diff = new filings. Table rows contain pipe delimiters.
ADDED_ROWS=$(echo "$DETAIL" | jq -r '
  .data.pages[]?.diff.text // empty' | grep -E '^\+.*\|.*\|' | grep -vE '^\+\s*\|\s*-+' || true)

if [ "$MEANINGFUL" = "0" ] || [ -z "$ADDED_ROWS" ]; then
  echo "Change was not a meaningful new insider-sale row (page churn). Exiting."
  exit 0
fi

echo "NEW FILING(S) DETECTED:"
echo "$ADDED_ROWS"
```

### Step 4 — Extract role + transaction amount for each new filing

Each added row is a markdown table row with these columns (pipe-delimited):

| # | Column | awk field (`-F'\|'`) |
|---|--------|----------------------|
| c1 | Transaction Date + type | `$2` |
| c2 | Reported DateTime | `$3` |
| c3 | Company | `$4` |
| c4 | Symbol | `$5` |
| c5 | **Insider Relationship** (name + role) | `$6` |
| c6 | Shares Traded | `$7` |
| c7 | Average Price | `$8` |
| c8 | **Total Amount** ($ value of the sale) | `$9` |
| c9 | Shares Owned | `$10` |
| c10 | Filing (`View` link) | `$11` |

The two fields the gates need — the **role** (c5) and the **Total Amount** (c8) — are in the
row for essentially every filing, so no extra fetch is normally required. **Firecrawl is the
fallback** when a row is missing or ambiguous (empty/garbled role, or a `$0`/blank Total
Amount): scrape the filing's `View` page (c10) or the insider's page and read the reporting
person's title and the transaction total from there —

```bash
# Fallback context fetch (only when the row itself is insufficient):
curl -s -X POST https://api.firecrawl.dev/v2/scrape -H "$AUTH" -H "Content-Type: application/json" \
  -d "{\"url\":\"$VIEW_LINK\",\"formats\":[\"markdown\"],\"onlyMainContent\":true}" | jq -r '.data.markdown'
```

### Step 5 — Qualifying gates: C-suite **AND** Total Amount > $1,000,000

Keep a filing **only if both** hold. **Interpretation: this is an AND** — "only interested in
sales > $1M" is a hard floor, so a $1M+ sale by a non-exec is dropped, and a C-suite sale
under $1M is dropped. (If you ever want OR instead, change the `&&` in the gate below to `||`.)

**C-suite** = the reporting person holds a chief-level executive title: any `Chief … Officer`
(CEO, CFO, CTO, COO, CIO, CMO, CLO, CAO, CHRO, CISO, CRO, CCO, …) or **President**. Excluded:
Director, 10% Owner, plain General Counsel, Secretary/Treasurer, and VP/EVP/SVP that isn't a
chief title. Roles are messy free text and rows can list several parties — **use judgment**,
and pull the filing via Firecrawl (Step 4) when a title is genuinely unclear. Passing the role
gate needs **any one** listed party to be C-suite.

```bash
MIN_USD=1000000
: > /tmp/qualified.jsonl

is_csuite() {  # role text -> exit 0 if C-suite
  local r; r=$(printf '%s' "$1" | tr 'A-Z' 'a-z')
  printf '%s' "$r" | grep -qE 'chief[a-z &.,/()-]*officer|\b(ceo|cfo|cto|coo|cio|cmo|clo|cao|chro|ciso|cro|cco|cdo|cgo)\b' && return 0
  printf '%s' "$r" | grep -qE '\bpresident\b' && ! printf '%s' "$r" | grep -qE 'vice[ -]president' && return 0
  return 1
}

printf '%s\n' "$ADDED_ROWS" | while IFS= read -r row; do
  [ -z "$row" ] && continue
  line=$(printf '%s' "$row" | sed 's/^+//')
  relationship=$(printf '%s' "$line" | awk -F'|' '{print $6}')
  company=$(printf '%s' "$line" | awk -F'|' '{print $4}' | sed -E 's/\[([^]]*)\].*/\1/; s/^ *//; s/ *$//')
  symbol=$(printf '%s' "$line"  | awk -F'|' '{print $5}' | sed -E 's/\[([^]]*)\].*/\1/; s/^ *//; s/ *$//')
  total_cell=$(printf '%s' "$line" | awk -F'|' '{print $9}')
  VIEW_LINK=$(printf '%s' "$line" | awk -F'|' '{print $11}' | grep -oE 'https?://[^) ]+' | head -1)

  # Total Amount -> integer USD (handles "$19,246,616", "$0", blanks)
  amount=$(printf '%s' "$total_cell" | grep -oE '[0-9][0-9,]*' | head -1 | tr -d ',')
  # Role text = relationship cell minus the [Name](link) parts (what's left is the title(s))
  role_text=$(printf '%s' "$relationship" | sed -E 's/\[[^]]*\]\([^)]*\)//g; s/<br>/ ; /g; s/^[ ;]*//; s/[ ;]*$//')

  # If role or amount is missing/ambiguous, resolve via Firecrawl (Step 4) before deciding.

  role_ok=no;  is_csuite "$role_text" && role_ok=yes
  amt_ok=no;   [ -n "$amount" ] && [ "$amount" -gt "$MIN_USD" ] && amt_ok=yes

  echo "• $symbol ($company) | role='$role_text' csuite=$role_ok | amount=\$${amount:-?} over1M=$amt_ok"
  if [ "$role_ok" = yes ] && [ "$amt_ok" = yes ]; then      # <-- AND gate (see note above)
    jq -nc --arg s "$symbol" --arg c "$company" --arg r "$role_text" \
           --argjson a "${amount:-0}" --arg v "$VIEW_LINK" \
      '{symbol:$s, company:$c, role:$r, amount_usd:$a, filing_url:$v}' >> /tmp/qualified.jsonl
  fi
done

if [ ! -s /tmp/qualified.jsonl ]; then
  echo "No new filing is BOTH C-suite AND > \$1,000,000. Exiting — stopping the routine."
  exit 0
fi

# ── ONE FILING PER RUN ─────────────────────────────────────────────────────
# Process only the single newest qualifying filing. Rows were scanned top-down
# and the page is sorted newest-first, so line 1 is the newest. Any extras are
# logged (not silently dropped) but NOT researched this run.
FILING=$(head -1 /tmp/qualified.jsonl)
EXTRA=$(( $(wc -l < /tmp/qualified.jsonl) - 1 ))
if [ "$EXTRA" -gt 0 ]; then
  echo "NOTE: $EXTRA more qualifying filing(s) this run — NOT processed (one-per-run). For the record:"
  tail -n +2 /tmp/qualified.jsonl
fi
export SYMBOL=$(printf '%s' "$FILING" | jq -r .symbol)
export COMPANY=$(printf '%s' "$FILING" | jq -r .company)
export ROLE=$(printf '%s' "$FILING" | jq -r .role)
export AMOUNT=$(printf '%s' "$FILING" | jq -r .amount_usd)
export FILING_URL=$(printf '%s' "$FILING" | jq -r .filing_url)
echo "PROCESSING THIS RUN → $SYMBOL ($COMPANY) | $ROLE | \$$AMOUNT | $FILING_URL"
```

Everything below operates on this **one** filing.

### Step 6 — Extract structured data from the Form 4 (Firecrawl Scrape + JSON)

Use `/v2/scrape` with an inline `json` format (synchronous structured extraction — no need for
the async `/v2/extract` job for a single page). This turns the messy Form 4 HTML into clean
fields, including the **transaction code** (`S` = open-market sale, `F` = tax-withholding, `M` =
option exercise, `G` = gift) and whether the sale was under a **Rule 10b5-1 plan** — both are
decisive for how bearish the signal actually is.

```bash
cat > /tmp/form4_req.json <<JSON
{
  "url": "$FILING_URL",
  "onlyMainContent": true,
  "formats": [{
    "type": "json",
    "prompt": "Extract this SEC Form 4 filing. Include the reporting person (name, CIK, address), the issuer (name, ticker, CIK), the person's relationship and title, EVERY transaction row (which table, security title, transaction date, transaction code, acquired vs disposed, share count, price per share, shares owned after, and direct/indirect ownership), all footnotes verbatim, whether any footnote references a Rule 10b5-1 trading plan, and the signature date.",
    "schema": {
      "type": "object",
      "properties": {
        "reporting_person": {"type":"object","properties":{"name":{"type":"string"},"cik":{"type":"string"},"address":{"type":"string"}}},
        "issuer": {"type":"object","properties":{"name":{"type":"string"},"ticker":{"type":"string"},"cik":{"type":"string"}}},
        "relationship": {"type":"string"},
        "transactions": {"type":"array","items":{"type":"object","properties":{
          "table":{"type":"string"},"security_title":{"type":"string"},"date":{"type":"string"},
          "transaction_code":{"type":"string"},"acquired_or_disposed":{"type":"string"},
          "shares":{"type":"number"},"price":{"type":"number"},
          "shares_owned_after":{"type":"number"},"ownership":{"type":"string"}}}},
        "footnotes": {"type":"array","items":{"type":"string"}},
        "rule_10b5_1_plan": {"type":"boolean"},
        "signature_date": {"type":"string"}
      }
    }
  }]
}
JSON
FORM4=$(curl -s -X POST https://api.firecrawl.dev/v2/scrape -H "$AUTH" -H "Content-Type: application/json" -d @/tmp/form4_req.json | jq '.data.json')
echo "$FORM4" | jq .
TICKER=$(echo "$FORM4" | jq -r '.issuer.ticker // env.SYMBOL')
PERSON=$(echo "$FORM4" | jq -r '.reporting_person.name // "the insider"')
```

If a field comes back null/empty, fall back to scraping the issuer or insider page (their
`secform4.com/insider-trading/<cik>.htm` links) for more context before continuing.

### Step 7 — Deep research (Firecrawl Search — you write the queries)

Firecrawl's standalone `/deep-research` endpoint is deprecated and `/research` is a papers
index — so **run the research yourself** as an iterative loop over `/v2/search`, which is the
production tool for this. You are the researcher: form hypotheses, write queries, read results,
follow the threads that matter, and search again. Aim for a well-rounded evidence base, not a
fixed number of calls.

Helper (search → web + news results with titles, urls, snippets):

```bash
fc_search() {  # fc_search "query" [sources_csv] [tbs]   e.g. fc_search "HAL stock outlook" web,news qdr:m
  local q="$1" src="${2:-web,news}" tbs="${3:-}"
  local sj; sj=$(printf '%s' "$src" | jq -Rc 'split(",")')
  jq -nc --arg q "$q" --argjson s "$sj" --arg tbs "$tbs" \
    '{query:$q, sources:$s, limit:6} + (if $tbs=="" then {} else {tbs:$tbs} end)' > /tmp/search_req.json
  # NOTE: /v2/search nests results under .data.{web,news}; web items use .description, news items use .snippet + .date
  curl -s -X POST https://api.firecrawl.dev/v2/search -H "$AUTH" -H "Content-Type: application/json" -d @/tmp/search_req.json \
    | jq -r '(.data.web//[])[]?  | "WEB  | \(.title) | \(.url)\n       \(.description // "")",
             (.data.news//[])[]? | "NEWS | \(.title) | \(.url) | \(.date // "")\n       \(.snippet // "")"'
}
```

Cover these axes (write your own queries; these are starting points — adapt to what you find):

- **The company** — `"$COMPANY" latest news`, earnings/guidance, downgrades, litigation, M&A, layoffs.
- **The stock** — `"$TICKER" stock price target analyst`, valuation, short interest, recent moves.
- **The individual** — `"$PERSON" "$COMPANY"` — tenure, departures, a *pattern* of prior sales, any cause for concern.
- **Industry / macro** — sector outlook and the specific drivers for this company (e.g. for an oilfield-services name: crude prices, rig counts, capex cycles), plus competitors.
- **Anything else** the filing surfaces (a footnote, a subsidiary, a co-filer) that could move the thesis.

Recency: **`tbs` (`qdr:d|w|m|y`) only filters `web` results, not `news`.** Use `sources:["news"]`
(already recency-ranked) for fresh headlines, and `web` + `tbs` for time-boxed background. When a
result looks pivotal, pull its full text with `/v2/scrape` (markdown) before relying on it.

```bash
# deep-read a pivotal source:
curl -s -X POST https://api.firecrawl.dev/v2/scrape -H "$AUTH" -H "Content-Type: application/json" \
  -d "{\"url\":\"<url>\",\"formats\":[\"markdown\"],\"onlyMainContent\":true}" | jq -r '.data.markdown'
```

### Step 8 — Synthesize the report and render it to a PDF

Write the analysis **yourself** from the Form 4 facts (Step 6) + your research (Step 7) as a single
JSON object at `/tmp/report.json`, then render it to a clean PDF with the repo's
`tools/render_report.py` (reportlab; the presentation is fixed so every report looks the same — you
own the content). The PDF leads with the **recommendation + one-liner + at-a-glance bullets**, then
the transaction facts, then the grounding sections — so the buy/sell call is readable in five seconds
and everything beneath it backs that call up.

`/tmp/report.json` shape (put an inline source URL next to each non-obvious claim in the section
bodies, and list every URL in `sources`):

```json
{
  "ticker": "<TICKER>", "company": "<COMPANY>", "generated_utc": "<UTC timestamp>",
  "recommendation": "BUY | SELL | HOLD | NEUTRAL | AVOID",
  "confidence": "Low | Medium | High",
  "one_liner": "One-sentence bottom line — the first thing the reader sees.",
  "summary_bullets": ["3–6 at-a-glance points that justify the call"],
  "transaction": {"person":"", "title":"", "date":"", "code":"S|F|M|G", "plan_10b5_1": true,
                  "shares":0, "price":0, "amount":0, "ownership":"Direct|Indirect",
                  "stake_context":"size vs remaining holdings"},
  "sections": [
    {"heading":"Signal read", "body":"How bearish/neutral and why: magnitude vs holdings, role, discretionary vs 10b5-1, lone seller vs cluster.", "sources":["url"]},
    {"heading":"Company", "body":"What it does; recent results, guidance, risks.", "sources":["url"]},
    {"heading":"Stock", "body":"Price action, valuation, analyst targets/sentiment.", "sources":["url"]},
    {"heading":"The individual", "body":"Role/tenure; any track record or pattern of prior sales.", "sources":["url"]},
    {"heading":"Industry & macro", "body":"Sector outlook and the drivers/competitors that bear on this name.", "sources":["url"]},
    {"heading":"Why this call", "body":"The 2–3 reasons driving the lean and the risks that would flip it."}
  ],
  "sources": ["every url relied on"],
  "caveats": "Timing mismatch: the public feed is ~6 months delayed, so this filing is historical while the research reflects today — say where that gap matters."
}
```

Section bodies support simple markdown (`**bold**`, `_italic_`, `- bullets`, blank-line paragraphs).
Set `recommendation` decisively — it drives the coloured banner (green BUY / red SELL / amber HOLD).

```bash
mkdir -p reports
python3 -m pip install --quiet reportlab 2>/dev/null || pip3 install --quiet reportlab 2>/dev/null || true
PDF="reports/$(echo "$FORM4" | jq -r '(.transactions[0].date // "undated")' | tr '/' '-')-${TICKER}-$(echo "$PERSON" | awk '{print $1}').pdf"
python3 tools/render_report.py /tmp/report.json "$PDF"
[ -s "$PDF" ] || { echo "PDF render failed — not delivering a broken file."; exit 1; }
echo "Rendered $PDF ($(wc -c < "$PDF") bytes)"
```

### Step 9 — Host the PDF and DM the signal to Slack

Host the PDF on **tmpfiles.org** (no auth) to get a shareable review/download link, then DM the DM
channel **`D99D96Y9Y`** (Lucas's DM in the CoinLedger workspace — the workspace the attached Slack
connector is authed to) via the Slack **`slack_send_message`** MCP tool. This needs **no bot token
and no `files:write`** — just the attached Slack connector. The message leads with the signal so it
is scannable straight from the DM list. This is delivery **instead of** committing — nothing is
committed to the repo.

```bash
# Upload to tmpfiles (expire=172800s = 48h max, so the link survives long enough to review)
UP=$(curl -s -X POST https://tmpfiles.org/api/v1/upload -F "file=@$PDF" -F "expire=172800")
LINK=$(echo "$UP" | jq -r '.data.url // empty')   # -> https://tmpfiles.org/<id>/<name>.pdf (a review+download page)
[ -z "$LINK" ] && echo "tmpfiles upload failed: $UP"
REC=$(jq -r .recommendation /tmp/report.json); CONF=$(jq -r .confidence /tmp/report.json)
echo "signal=$REC conf=$CONF link=$LINK"
```

Then call the **`slack_send_message`** MCP tool with `channel_id = "D99D96Y9Y"` and a markdown
`message` that contains, **in this order**:

1. **The signal, unmistakably** — the call with an emoji cue (🟢 `BUY` · 🔴 `SELL` · 🟡 `HOLD` /
   `NEUTRAL`) and the ticker, e.g. `🟡 *HOLD — $HAL (Halliburton)*`.
2. **Confidence** — e.g. `Confidence: *Medium*`.
3. **One line of what happened** — who sold how much, when (from `transaction`).
4. **3–4 quick “why” bullets** — the reasons driving the call (from `summary_bullets`).
5. **The link** — a labelled markdown link so it reads nicely, e.g.
   `📄 [Review & download the full report](<LINK>)`. If unsure Slack will render the label, put the
   bare `$LINK` on its own line — a bare URL is always clickable.
6. **Footer** — `_Link expires in ~48h · automated research, not financial advice_`.

Example `message` body:

```
🟡 *HOLD — $HAL (Halliburton)*   ·   Confidence: *Medium*

Jeffrey Slocum (EVP & COO) sold $771,808 on 2026-01-09.

*Why:*
• Pre-planned 10b5-1 sale (plan set Aug 2025) — mechanical, not a discretionary bearish call.
• Small & partial — ~11% of his stake trimmed; ~187K shares retained.
• Newly-promoted COO (effective Jan 1 2026), not a departure.
• Backdrop mildly constructive: Q1 2026 beat, Moderate-Buy consensus, low-$40s targets.

📄 [Review & download the full report](https://tmpfiles.org/…/2026-01-09-hal-slocum.pdf)
_Link expires in ~48h · automated research, not financial advice_
```

If the tmpfiles upload failed, still send the DM with the signal + confidence + bullets and note the
report link was unavailable this run.

## Prototype caveats

- **Data is ~6 months delayed.** We monitor the public (unauthenticated) page, which
  `secform4.com` delays ~6 months for non-subscribers. New rows are real Form 4 sales but
  old ones, and the delayed feed likely advances ~once per trading day. Real-time detection
  needs an authenticated Insider Pro session — deferred.
- **You fire hourly regardless of change.** The Step 2/3 gates are what keep the ~23/24
  no-op fires cheap. Never skip them.
- A single-page monitor's first-ever check is `status: "new"` (baseline). Real new filings
  arrive as `status: "changed"` — which is why the gate keys on `summary.changed`, not `new`.
