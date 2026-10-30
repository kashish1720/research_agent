# CLI Demo Video Script

Target length: about 5 minutes. Speak slowly; pause the recording while a live command runs.
The evaluation table and saved JSON are evidence of recorded runs, not a promise that a new
live run will return the same output.

## Main demo (about 5 minutes)

| Time | Show on screen | Exact command or file to open | Spoken text |
|---|---|---|---|
| 00:00–00:20 | Terminal at the repository root; title card: “Research Agent CLI” | `Get-Location` | “Hi, I’m presenting my research-agent CLI. I built it with help from an AI coding agent; I made the design decisions, ran the evaluations, and debugged failures. I’ll show how the workflow handles evidence and failures, not claim that its answers are always correct.” |
| 00:20–00:55 | README’s Mermaid flowchart and its workflow steps | Open `README.md`, sections 6 and 7 | “The LLM plans the sub-questions and writes search queries, then proposes evidence passages and drafts an answer. Python controls the stages, budgets, retries, URL safety, and when to stop. The evidence table gives synthesis short passages and IDs to cite, rather than raw pages.” |
| 00:55–01:40 | Run the comparison, then show the final answer and trace | `cls; $env:USE_CACHE="false"; python main.py --trace "Compare the top 3 open-source vector databases for a startup building RAG products."` | “I’m disabling the development cache so this run uses the configured tools instead of replaying a cached result. This question asks for a comparison of vector databases. I’ll let the command finish, then inspect the answer and the per-node trace.” |
| 01:40–02:05 | Scroll through “Short answer,” “Key findings,” “Sources used,” “Confidence,” “Limitations and assumptions,” “Run info,” and the trace table | Open `src/research_agent/render.py`; then point to the just-rendered terminal output | “These are the renderer’s real section names. Findings show their evidence IDs, and sources show the source details; confidence includes its reasons. Run info gives status and usage, while the trace table summarizes time, tokens, tool calls, retries, and failures by node.” |
| 02:05–02:28 | Show the verbatim-check function and a test that rejects a paraphrase | Open `src/research_agent/nodes/extract.py` at `passage_is_verbatim`; open `tests/test_extract_node.py` at `test_paraphrase_is_not_accepted_as_verbatim` | “The function `passage_is_verbatim` lives in `src/research_agent/nodes/extract.py`. It checks whether normalized passage text occurs contiguously in normalized fetched text. That proves the words are present on the page; it does not prove that the claim is true or that the passage is in context.” |
| 02:28–02:42 | Run the nonsense question and show its status and zero search/fetch usage | `cls; $env:USE_CACHE="false"; python main.py --trace "asdkjh qwpoeiru zxcmnb lkjhg"` | “Here is a deliberately unclear query. The saved evaluation for this input is `FAILED_GRACEFULLY`, with no findings, searches, or fetches. A new run can vary, so I’ll describe the output that actually appears rather than promise an outcome.” |
| 02:42–03:05 | Run the seeded timeout case; show reasons and trace retries/failures | `$env:USE_CACHE="false"; $env:FAULT_INJECT="timeout:0.6"; python main.py --trace "Compare the top 3 open-source vector databases for a startup building RAG products."; Remove-Item Env:FAULT_INJECT` | “This run enables the timeout fault injector. The trace is where I inspect retries and failures, and the final status tells me whether the run degraded. The saved evaluation for this fault case is `DEGRADED`; a fresh run need not match it exactly.” |
| 03:05–03:25 | Run the offline injection tests and show their names in the test output | `cls; pytest -q -k "canary or injection" -v` | “These are offline tests, so they do not call the LLM or the network. The named checks include `test_injection_canary_is_removed_before_extraction_and_final_json`, `test_injection_in_fetched_page_is_removed_and_flagged`, and `test_heavy_injection_drops_source_and_traces_note`. They test removal, flagging, and dropping of hostile page content.” |
| 03:25–03:50 | Show the NAT64 commit, DNS diagnostic script, and URL-safety redirect test | `git show --stat 1f4f5af`; then open `scripts/diag_dns.py` and `tests/test_url_safety.py` at `test_redirect_to_private_nat64_host_is_blocked_before_following` | “During live testing I found an SSRF validation issue involving public hosts resolving to NAT64 addresses. Git history records a commit titled `nat64 fix`; the diagnostic script displays resolved addresses and their `is_global` property. The tests cover unwrapping public NAT64 addresses, rejecting private embedded addresses, and blocking a redirect to a private NAT64 address before following it.” |
| 03:50–04:15 | Open `evals/results.md`; point to the PASS column and table | Open `evals/results.md` | “This table records five evaluation cases: vector database, multi-agent, memory approaches, nonsense input, and injected timeouts. The recorded statuses are `DONE`, `FAILED_GRACEFULLY`, or `DEGRADED`, as shown. PASS only means the final status was in that case’s expected list; it does not measure answer correctness.” |
| 04:15–04:35 | Show README’s current limitations and evaluation gaps | Open `README.md`, sections 16 and 19 | “The evaluation did not include human claim review, a no-tools baseline, or an unsupported-claim rate. Other limitations include unsupported PDFs and JavaScript-heavy pages, DNS-rebinding time-of-check/time-of-use risk, and confidence that can overstate independence when sites repeat one source. These are reasons to treat the output as research assistance, not proof.” |
| 04:35–04:55 | Return to the README future-improvements list; finish on the terminal prompt | Open `README.md`, section 19 | “My next steps are to persist evidence so runs can be audited, add a no-tools baseline, and improve short-answer verification. I’d also want more human review before making stronger claims about answer quality. Thanks for watching.” |

## Do not say

- Do not claim that the system guarantees factual accuracy or that every answer is correct.
- Do not call pipeline support markers independent human verification.
- Do not describe PASS as proof of answer quality; it only records that the final status matched the expected-status list.
- Do not invent benchmark numbers, unseen live output, or results that are not in the repository.
- Do not imply that the demo’s new live runs will reproduce the saved evaluation statuses or timing.
- Do not expose `.env`, API keys, or other credentials on screen.

## Pre-recording checklist

- Set a readable terminal font and size; confirm the repository root is the working directory.
- Use `cls` between commands so previous output does not distract.
- Keep `.env` closed and out of the recording; never show credentials.
- Set `$env:USE_CACHE="false"` before each live run so the cache does not replay a prior result.
- Wait 60 seconds between live runs because of the free-tier tokens-per-minute limit.
- Open the cited files and saved evaluation artifacts in advance; do not edit them during the recording.
- Have the exact prompt-injection test command ready. It is an offline test command.

## Record the screen and add voiceover on a Windows laptop

These steps use Windows Xbox Game Bar to record the VS Code window, then Clipchamp to add
or adjust narration. Button names can vary slightly by Windows or Clipchamp version.

### A. Prepare the screen

1. Plug in the laptop and connect headphones if available, so speaker audio does not leak into the microphone.
2. Open **Start**, type **Sound settings**, and open it. Under **Input**, select the microphone you will use and speak; check thawt the input meter moves.
3. Open VS Code and open the project folder. Use the integrated terminal so the terminal, code, and results stay inside the VS Code window.
4. Arrange the editor and terminal before recording. Close unrelated tabs and panels, enlarge the text, and make sure no `.env` file or credential is visible.
5. In the VS Code terminal, run `Get-Location` to check the repository folder. Do not start the demo's live commands until you are ready to record.

### B. Record the demo with narration in one take

1. Click the VS Code window so it is active.
2. Press **Windows key + G** to open Xbox Game Bar. If asked whether this is a game, confirm that it is an app you want to record.
3. Open the **Capture** widget. In its settings, choose the microphone you tested and turn microphone recording on. The shortcut **Windows key + Alt + M** toggles the microphone during capture.
4. Close overlays that cover the content, while leaving VS Code active. Say a brief test sentence and confirm the microphone indicator is on.
5. Press **Windows key + Alt + R** to start recording. Wait a moment, then begin the intro and follow the timed script above.
6. To keep pauses tidy, stop the recording with **Windows key + Alt + R** while a long command is running. Start a new clip when ready to continue; alternatively, leave the recording running and trim the pause later.
7. When finished, press **Windows key + Alt + R** again. The recording is usually saved under **Videos > Captures**; open Game Bar's **Gallery** if you need to locate it.
8. Play the clip once and check that both the screen and voice are present and understandable.

### C. Add or replace the voiceover in Clipchamp

1. Open **Start**, type **Clipchamp**, and open the editor. Choose **Create a new video**.
2. Choose **Import media**, select the recorded video from **Videos > Captures**, then drag it onto the timeline.
3. To add a separate narration track, open **Record & create** and choose the audio or voice-recording option. Select the tested microphone, allow microphone access if Windows asks, and record while watching the video.
4. Add the recorded narration to an audio track below the video. Drag it so the first spoken words align with the start of the demo; split or trim sections to line up later narration.
5. If the video already contains narration that you do not want, select its audio in the timeline and mute or detach/delete that audio before adding the new voiceover. Keep the screen video.
6. Play the full timeline. Check that speech is audible, commands remain readable, and the video does not show secrets. Adjust the clip volume if needed.
7. Choose **Export**, select the available 1080p option if offered, and save the finished video. Reopen the exported file and check it before uploading.

If Xbox Game Bar is unavailable or blocks the window you need, use another screen recorder already installed on the laptop, select the VS Code window, enable microphone capture, and follow the same review and export steps. Do not install or sign in to a new service just for this recording unless you choose to.

## Fallback plan for a failed run or HTTP 429

Let the configured retries finish; do not raise limits or immediately repeat the run. If it still fails or returns 429, say that the live attempt did not complete successfully, show the failure and trace without presenting them as a successful answer, and move on. If a completed example is needed, open the matching saved file under `samples/` or the recorded table in `evals/results.md`, and label it clearly as saved evaluation output rather than the current run.

## 90-second shortened version

| Time | Show / command | Spoken text |
|---|---|---|
| 00:00–00:15 | README Mermaid architecture | “I built this CLI with help from an AI coding agent; I made the design decisions, ran evaluations, and debugged failures. The LLM plans and writes queries; Python controls flow, budgets, retries, and safety.” |
| 00:15–00:40 | `$env:USE_CACHE="false"; python main.py --trace "Compare the top 3 open-source vector databases for a startup building RAG products."` | “This is the comparison example with the development cache disabled. I’ll point out the answer, cited findings, sources, confidence reasons, limitations, run information, and node trace.” |
| 00:40–00:55 | Open `src/research_agent/nodes/extract.py` at `passage_is_verbatim` | “This function checks that a normalized passage occurs in the fetched text. It proves presence of those words, not that the claim is true or in context.” |
| 00:55–01:15 | Open `evals/results.md`; point to `nonsense` and `search_faults` | “The saved table includes a graceful failure for nonsense input and a degraded timeout case. PASS means only that the final status was expected, not that the answer was correct.” |
| 01:15–01:30 | Open README limitations / show test command | “The project still lacks human review and a no-tools baseline, and it has known SSRF and content-coverage limitations. I’m treating these as follow-up work, not claiming they are solved.” |

## Interviewer questions

### 1. Why use a bounded agentic workflow?

The LLM plans the research and writes queries, while Python controls stage order, budgets, retries, and stopping. This makes the tool use and failure handling explicit instead of relying on one unconstrained completion.

### 2. How does the agent avoid searching nonsense input?

The saved `nonsense` sample is `FAILED_GRACEFULLY` and reports zero searches and zero fetches. It returns a clarification instead of research findings.

### 3. What does `passage_is_verbatim` verify?

It normalizes a proposed passage and the source text, then checks for a contiguous occurrence. This verifies text presence, not truth or context.

### 4. How are source citations connected to findings?

Findings cite evidence IDs assigned by the code. The verifier drops findings with evidence IDs that are empty or do not resolve to evidence in the run.

### 5. What does a `supported` marker mean?

It is the pipeline’s own entailment-check result for the finding and its cited passages. It is not an independent human judgment or a guarantee of correctness.

### 6. How does the project handle prompt injection?

The page text is treated as untrusted, known instruction-like lines are scanned and removed, and the extractor has no tools. Offline tests check that injected content is removed or flagged and that a canary does not appear in the final answer.

### 7. What changed in the NAT64 fix?

The URL-safety code unwraps NAT64 and IPv4-mapped IPv6 addresses before applying IPv4 address checks. Tests verify public NAT64 handling, rejection of private embedded addresses, and blocking of a redirect to a private NAT64 address.

### 8. What does the evaluation PASS result mean?

It means the final status was among the expected statuses for that case. The evaluation does not measure answer correctness or an unsupported-claim rate.
