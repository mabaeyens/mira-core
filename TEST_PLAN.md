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
| 4.1 | While a response is streaming, observe Send button | Send button hidden; red Stop button visible in its place |
| 4.2 | While streaming, observe textarea | Disabled; cannot type |
| 4.3 | While streaming, observe paperclip button | Disabled |
| 4.4 | After response completes | Stop button hidden; Send button and textarea re-enabled |

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

## 11. RAG (large document retrieval)

> **Test documents needed:**
> - `small.pdf` — a text-based PDF under 80k chars (e.g. a 5-page article)
> - `large.pdf` — a text-based PDF over 80k chars with specific facts in the latter half
> - `second.pdf` — a second document on a different topic
> - `scanned.pdf` — an image-only PDF with no text layer
>
> **Accuracy ground truth:** before running these tests, read `large.pdf` and write down 3 specific facts that appear only after the first ~80k characters. These are your retrieval accuracy probes.

### 11a. Indexing and UI feedback

| # | Action | Expected |
|---|--------|----------|
| 11.1 | Attach any PDF (small or large) | Indexing chip appears: "Indexing [filename]…" with spinner |
| 11.2 | Wait for indexing to complete | Chip updates: "Indexed N chunks — [filename]" |
| 11.3 | Attach a second PDF while the first is indexed | Second doc indexed and added; first doc remains in index |
| 11.4 | Observe the Documents panel in the web UI | Both filenames listed with ✕ remove buttons |
| 11.5 | Click ✕ on one document | That document removed from index; remaining document still listed |
| 11.6 | Click Reset | Documents panel disappears; RAG index cleared along with conversation history |

### 11b. Retrieval accuracy

| # | Action | Expected |
|---|--------|----------|
| 11.7 | Attach `large.pdf`; ask about a fact from the **first** quarter of the document | Correct answer; model cites the right section |
| 11.8 | Ask about a fact from the **latter half** of the document (past 80k chars) | Correct answer — proves RAG retrieves beyond the old truncation point |
| 11.9 | Ask one of your 3 ground-truth probe questions | Answer matches the known fact exactly |
| 11.10 | Ask a second probe question without re-attaching | Correct answer — RAG index persists across turns |
| 11.11 | Ask a question whose answer is not in the document | Model says it cannot find that information in the document (does not hallucinate from the document) |
| 11.12 | Ask a question entirely unrelated to any indexed document | Model answers from general knowledge; no RAG chunks injected (reranker score below threshold) |

### 11c. Reranking quality

| # | Action | Expected |
|---|--------|----------|
| 11.13 | Ask a vague question with several plausible sections in the document | Answer uses the most relevant section, not just the first retrieved chunk |
| 11.14 | Ask the same question with reranking disabled (set `RERANK_TOP_K = 0` temporarily) | Note any difference in answer quality — confirms reranker is adding value |

### 11d. Multi-document retrieval

| # | Action | Expected |
|---|--------|----------|
| 11.15 | Index `large.pdf` and `second.pdf` (different topics); ask about topic from `large.pdf` | Correct answer; chunks from `second.pdf` not injected |
| 11.16 | Ask about topic from `second.pdf` | Correct answer from `second.pdf` |
| 11.17 | Ask a question that spans both documents | Answer draws from both; no hallucination |

### 11e. Edge cases

| # | Action | Expected |
|---|--------|----------|
| 11.18 | Attach `scanned.pdf` | Warning chip: scanned PDF, no text extracted; no indexing attempted |
| 11.19 | Attach `small.pdf` (under 80k chars) | Still goes through RAG (PDFs always use RAG); indexing chip shown |
| 11.20 | Index enough documents to exceed the chunk threshold | Warning chip recommending the user unload documents |

---

## 12. RAG chunk injection panel

> **Setup:** use the same `large.pdf` from section 11. The panel only appears when the RAG index is non-empty and the model call completes (not during streaming).

### 12a. Panel appearance

| # | Action | Expected |
|---|--------|----------|
| 12.1 | Attach a PDF; ask a specific question about its content | Below the assistant bubble, a green "📚 N document section(s) used ▴" chip appears |
| 12.2 | Observe chip position | Panel sits below the answer bubble, not inside it, and not above it |
| 12.3 | Observe chip state by default | Expanded (open); source rows visible immediately without clicking |
| 12.4 | Observe each row in the expanded list | Each row shows: document filename in bold, CrossEncoder score (e.g. `· score: 1.42`), and italic preview text |
| 12.5 | Ask a question unrelated to any indexed document | No RAG panel appears (reranker filtered all chunks; `rag_context` event not emitted) |

### 12b. Collapse / expand

| # | Action | Expected |
|---|--------|----------|
| 12.6 | Click the green chip | Panel collapses; arrow changes from ▴ to ▾ |
| 12.7 | Click the chip again | Panel re-expands; arrow returns to ▴ |
| 12.8 | Scroll the chat while panel is collapsed | Chip stays collapsed (state not reset by scroll) |

### 12c. Multiple turns

| # | Action | Expected |
|---|--------|----------|
| 12.9 | Ask a second question about the same document (no re-attach) | Second answer also shows a RAG panel with its own chunks (may differ from turn 1) |
| 12.10 | Compare chunk previews across two turns on different topics | Different source chunks surfaced — confirms per-question retrieval, not a cached result |
| 12.11 | Collapse the panel from turn 1; send a new message | Turn 1 panel remains collapsed; turn 2 panel starts expanded independently |

### 12d. Score and content accuracy

| # | Action | Expected |
|---|--------|----------|
| 12.12 | Observe scores in the panel for a highly relevant question | Scores are positive and > 0.0; higher-scoring chunks ranked first |
| 12.13 | Read the preview text for the top chunk | Preview text is recognisably relevant to the question asked |
| 12.14 | Ask a meta-instruction query ("Summarise this") immediately after attaching a PDF | RAG panel appears (score threshold bypassed for same-turn attach); chunks shown even if scores are low |

### 12e. Multi-document index

| # | Action | Expected |
|---|--------|----------|
| 12.15 | Index two PDFs on different topics; ask about topic A | Panel shows chunks only from document A (or mostly A by score) |
| 12.16 | Ask about topic B | Panel shows chunks from document B |
| 12.17 | Remove document A via the Documents panel; ask about topic A | No RAG panel (or empty panel) — chunks from A no longer in index |

---

## 13. Clickable sources and fetch chips

### 13a. Search sources

| # | Action | Expected |
|---|--------|----------|
| 13.1 | Ask a current-events question that triggers a search | After search completes, a sources list appears below the search chip with bulleted links |
| 13.2 | Observe sources list default state | List is expanded (open) by default; ▴ indicator on the chip |
| 13.3 | Click the search chip | Sources list collapses; indicator changes to ▾ |
| 13.4 | Click the chip again | Sources list re-expands; indicator returns to ▴ |
| 13.5 | Click a link in the sources list | Opens the source URL in a new browser tab |

### 13b. Fetch chips (inline, during streaming)

| # | Action | Expected |
|---|--------|----------|
| 13.6 | Ask a question where the model fetches a page (`fetch_url`) | A "Reading: hostname" chip with spinner appears, then updates to "✅ Read N.Nk chars — hostname" with a clickable link |
| 13.7 | Click the hostname link in the fetch chip | Opens the fetched URL in a new tab |
| 13.8 | Ask a question requiring multiple searches or fetches | Each search and fetch produces its own chip; all remain visible after the response |

### 13c. Fetch context panel (after answer)

> The "pages read" panel appears **below the answer bubble** after the turn completes, distinct from the inline fetch chip shown during streaming.

| # | Action | Expected |
|---|--------|----------|
| 13.9 | Ask a question that causes the model to fetch a page | Below the answer bubble, a blue "🌐 1 page read ▴" chip appears |
| 13.10 | Observe panel default state | Expanded by default; entry shows hostname link, char count, and italic preview text |
| 13.11 | Observe the hostname in the entry | Clickable link; opens the fetched URL in a new tab |
| 13.12 | Observe the preview text | First ~300 chars of the fetched page content, ending in "…" if truncated |
| 13.13 | Click the blue chip | Panel collapses; arrow changes to ▾ |
| 13.14 | Click the chip again | Panel re-expands; arrow returns to ▴ |
| 13.15 | Ask a question answered from general knowledge (no fetch) | No "pages read" panel appears |
| 13.16 | Ask a question that triggers two separate `fetch_url` calls | Panel reads "🌐 2 pages read ▴"; both entries visible with their respective hostnames and previews |
| 13.17 | Compare the inline fetch chip (during streaming) with the context panel | Chip shows hostname and char count live; panel shows those plus the preview text, after the answer is complete |
| 13.18 | Collapse the panel from turn 1; send a new message that also fetches | Turn 1 panel stays collapsed; turn 2 panel starts expanded independently |

---

## 14. Loading files from disk by path

> Tests the 📂 button (web) and `/attach` command (CLI). Both use `file_handler.load_file()` and produce the same attachment pipeline as uploaded files. The server must be able to read the given path — paths are resolved on the server's filesystem.

### 14a. Web — path input UI

| # | Action | Expected |
|---|--------|----------|
| 14.1 | Click the 📂 button | A path input row appears above the input area with a text field, "Add" button, and "Cancel" button |
| 14.2 | Press Cancel | Path input row hides; nothing staged |
| 14.3 | Type a valid absolute path and press Enter | Path input row hides; a blue 📂 path chip appears in the chips row |
| 14.4 | Type a valid absolute path and click Add | Same as 14.3 |
| 14.5 | Add two different paths | Both path chips shown |
| 14.6 | Click ✕ on a path chip | That chip removed; other chips unaffected |
| 14.7 | Path chip is visible in the user bubble after sending | Blue 📂 chip showing the full path appears at the top of the user message, same as file chips |
| 14.8 | Click 📂 while streaming | Button is disabled; path input row does not open |

### 14b. Web — file loading behaviour

| # | Action | Expected |
|---|--------|----------|
| 14.9 | Add a path to a text-based PDF; ask about its content | Indexing chip appears; RAG panel shown below answer with chunks from that file |
| 14.10 | Add a path to a `.txt` file; ask a question about it | File content injected inline; model answers correctly |
| 14.11 | Add a path to a `.py` or `.json` file | Content injected; model can reason about the code or data |
| 14.12 | Add a path to an image file | Model describes the image (multimodal path) |
| 14.13 | Add a path that does not exist | Amber ⚠️ warning chip: "Could not load '/path/…': File not found" |
| 14.14 | Mix a valid upload (📎) with a valid path (📂) in the same message | Both files processed; both chips shown in user bubble; model has access to both |
| 14.15 | Add a path to a large text file (> 80k chars) | Upgraded to RAG automatically; indexing chip shown |

### 14c. CLI — `/attach` command (existing, confirmed working)

| # | Action | Expected |
|---|--------|----------|
| 14.16 | `/attach /absolute/path/to/file.pdf` | `Attached: file.pdf (rag)` shown; file staged for next message |
| 14.17 | `/attach ~/relative/path/file.txt` | Tilde expanded; file loaded; `Attached: file.txt (text)` shown |
| 14.18 | `/attach /nonexistent/path.pdf` | Error message: "File not found: …" |
| 14.19 | `/files` after attaching two files | Both filenames and types listed |
| 14.20 | `/detach` clears path-attached files | No files staged; prompt returns to `You:` |
| 14.21 | Attach a PDF via `/attach`; ask a question | Indexing spinner shown; RAG chunks used; verbose mode shows chunk previews |

---

## 15. Stop button

| # | Action | Expected |
|---|--------|----------|
| 15.1 | Send any message; observe footer while streaming | Send button hidden; red Stop button visible in its place |
| 15.2 | After response completes | Stop button hidden; Send button returns |
| 15.3 | Send a message; click Stop during the thinking phase | Response aborts; thinking indicator removed; Stop disappears and Send returns |
| 15.4 | Send a message; click Stop while tokens are streaming | Streaming halts mid-sentence; partial text remains visible; Send re-enabled |
| 15.5 | Send a message; click Stop while a search chip is spinning | Search halts; chip may remain; Send re-enabled |
| 15.6 | After stopping, send a new message | Conversation continues normally; model has no memory of the cancelled turn |
| 15.7 | After stopping, send a follow-up that references the cancelled turn | Model has no context from the aborted turn (history was rolled back) |
