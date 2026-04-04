# Web UI Test Plan

**URL:** `http://localhost:8000`  
**Start server:** `python server.py`  
**Prerequisites:** Ollama running locally with `gemma4:26b` pulled

Test cases are grouped by feature area. Each case lists the action, expected result, and pass/fail column. Run top-to-bottom in a fresh browser tab (no prior session state) unless the case says otherwise.

---

## 1. Page load and initial state

| # | Action | Expected |
|---|--------|----------|
| 1.1 | Open `http://localhost:8000` | Page loads; header shows "ollama Search Tool" and model badge `gemma4:26b` |
| 1.2 | Inspect header controls | Verbose checkbox unchecked; Reset button visible |
| 1.3 | Inspect footer | Paperclip (📎) button and textarea visible; no file chips |
| 1.4 | Inspect chat area | Welcome screen with search icon and "Ask anything" message |

---

## 2. Basic chat — no search, no files

| # | Action | Expected |
|---|--------|----------|
| 2.1 | Type "What is 2 + 2?" and press Enter | Thinking indicator (three dots) appears; then response streams in; answer contains "4" |
| 2.2 | Observe answer rendering | Response rendered as markdown (not raw text) |
| 2.3 | Send a second message in the same session | Previous messages remain visible; conversation context is maintained |
| 2.4 | Press Shift+Enter in the textarea | Newline inserted; message not sent |
| 2.5 | Click Send with empty textarea | Nothing happens; no request sent |

---

## 3. Web search

| # | Action | Expected |
|---|--------|----------|
| 3.1 | Ask "Who won the most recent Champions League?" | Thinking → search chip with spinner and query text → "Found N results" chip → thinking → answer |
| 3.2 | Observe search chip final state | Chip shows ✅ and result count |
| 3.3 | Ask a question with no recent information ("What is the capital of France?") | Answer arrives without any search chip |
| 3.4 | Enable Verbose checkbox; ask a current-events question | Behaviour unchanged (verbose only affects CLI; web always shows search chips regardless) |

---

## 4. Streaming and UI state during response

| # | Action | Expected |
|---|--------|----------|
| 4.1 | While a response is streaming, observe Send button | Button is disabled (dimmed) |
| 4.2 | While streaming, observe textarea | Disabled; cannot type |
| 4.3 | While streaming, observe paperclip button | Disabled |
| 4.4 | After response completes | Send button and textarea re-enabled |

---

## 5. Reset

| # | Action | Expected |
|---|--------|----------|
| 5.1 | Send two messages; click Reset | Chat area clears; welcome screen returns |
| 5.2 | Send a follow-up message that references the earlier conversation | Model has no memory of the cleared messages |
| 5.3 | Click Reset while streaming | Button does nothing (disabled by isStreaming guard) — verify no crash |

---

## 6. File attachments — images

| # | Action | Expected |
|---|--------|----------|
| 6.1 | Click 📎; select a PNG or JPG image | File chip with filename appears above textarea |
| 6.2 | Click the ✕ on the chip | Chip removed; no files staged |
| 6.3 | Attach an image; type "Describe this image" and send | Image chip shown in user bubble; model response describes the image content |
| 6.4 | Send the next message (no attachment) | No file chip in user bubble; model answers from conversation context only |

---

## 7. File attachments — text and code

| # | Action | Expected |
|---|--------|----------|
| 7.1 | Attach a `.txt` or `.py` file; ask "Summarise this file" | Model response summarises the file content |
| 7.2 | Attach a `.json` file; ask "What keys are in the root object?" | Model correctly reads the JSON structure |
| 7.3 | Attach multiple files (e.g. two `.txt` files) | Both chips shown in footer and in user bubble; model can reference both |

---

## 8. File attachments — PDF

| # | Action | Expected |
|---|--------|----------|
| 8.1 | Attach a small text-based PDF (< 80k chars); ask a question about its content | Model answers from the document; no warning chip |
| 8.2 | Attach a large text-based PDF (> 80k chars) | Amber ⚠️ warning chip: "truncated to 80,000 characters … consider RAG" |
| 8.3 | Attach a scanned (image-only) PDF | Amber ⚠️ warning chip: "scanned PDF with no extractable text" |

---

## 9. File attachments — HTML

| # | Action | Expected |
|---|--------|----------|
| 9.1 | Attach an `.html` file; ask "What is the main topic of this page?" | Model answers from the readable text; no raw HTML tags in the injected content |

---

## 10. Error states

| # | Action | Expected |
|---|--------|----------|
| 10.1 | Stop Ollama; send a message | Red ❌ error chip with a descriptive message |
| 10.2 | Restart Ollama; send a message | Normal response; no residual error state |

---

## 11. Phase 3 — RAG for large PDFs *(to be tested after implementation)*

| # | Action | Expected |
|---|--------|----------|
| 11.1 | Attach a large PDF (> 80k chars) | Instead of a truncation warning, an indexing chip appears: "Indexing document…" |
| 11.2 | Observe indexing completion | Chip updates: "Indexed N chunks — [filename]" |
| 11.3 | Ask a specific question answered in the latter half of the document | Model answers correctly using retrieved chunks, not just the first 80k chars |
| 11.4 | Ask a follow-up question about the same document (no re-attachment) | RAG index is still active; model answers from retrieved chunks |
| 11.5 | Attach a second large PDF | Previous index replaced; new document indexed |
| 11.6 | Ask a question unrelated to the indexed document | Model answers from general knowledge or web search; RAG chunks not injected |
| 11.7 | Click Reset | RAG index cleared along with conversation history |
