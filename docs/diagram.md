# Architecture Diagrams

Mermaid diagrams for the Mira system. Render in any Markdown viewer that supports Mermaid (GitHub, VS Code with extension, etc.).

## System Overview

```mermaid
graph TD
    subgraph mira-apps["mira-apps (Swift/SwiftUI)"]
        MAC["macOS App\nMacRootView"]
        IOS["iOS App\niOSConnectedView"]
        VM["ChatViewModel\n@Observable @MainActor"]
        API["APIClient\nREST wrapper"]
        SSE["SSEClient\nAsyncThrowingStream"]

        MAC --> VM
        IOS --> VM
        VM --> API
        VM --> SSE
    end

    subgraph mira-core["mira-core (Python/FastAPI)"]
        SRV["server.py\nFastAPI + SSE"]
        ORCH["ChatOrchestrator\nstream_chat()"]
        RAG["RagEngine\nChromaDB + reranker"]
        SEARCH["SearchEngine\nBrave / DuckDuckGo / native"]
        FETCH["url_fetcher\nBeautifulSoup + Jina fallback"]
        FS["fs_tools / shell_tools\nworkspace-sandboxed"]
        GH["github_tools\nGitHub API"]
        DB["db.py\nSQLite"]
        FH["file_handler\nPDF/HTML/image/text"]

        SRV --> ORCH
        ORCH --> RAG
        ORCH --> SEARCH
        ORCH --> FETCH
        ORCH --> FS
        ORCH --> GH
        ORCH --> DB
        ORCH --> FH
    end

    subgraph infra["Local Infrastructure"]
        MLX["oMLX (port 8080)\nQwen3.6-35B-A3B\n(mlx-lm / dFlash compatible API)"]
        ST["SentenceTransformer\nnomic-embed-text-v1.5\n(local, 768 dims)"]
        OLLAMA["Ollama (port 11434)\n(optional inference backend)"]
        WEB["Web search\nBrave / DuckDuckGo / native\n+ Jina fetch fallback"]
        SQLITE[("SQLite\n~/.local/share/mira/")]
    end

    API -->|"REST\nHTTP/HTTPS"| SRV
    SSE -->|"SSE stream\nPOST /chat"| SRV
    ORCH -->|"chat + tool calling"| MLX
    RAG -->|"embeddings"| ST
    SEARCH -->|"web queries"| WEB
    FETCH -->|"page content"| WEB
    DB --> SQLITE
```

## Turn Lifecycle (SSE event flow)

```mermaid
sequenceDiagram
    participant Client as iOS/macOS/Web Client
    participant Server as server.py
    participant Orch as ChatOrchestrator
    participant Ollama as oMLX (Qwen3.6-35B)
    participant Tools as Tools (search/fetch/files/shell)

    Client->>Server: POST /chat (multipart: message + files)
    Server->>Server: clear cancel_event
    Server->>Server: spawn produce() thread
    Server-->>Client: SSE stream opened

    Orch->>Orch: index attachments (RAG)
    Server-->>Client: rag_indexing, rag_done

    Orch->>Ollama: stream chat (tool calling enabled)
    Server-->>Client: thinking

    loop Tool calls (up to MAX_TOOL_STEPS=10)
        Ollama-->>Orch: tool_call chunk
        Orch->>Tools: dispatch tool
        Server-->>Client: search_start / fetch_start
        Tools-->>Orch: result
        Server-->>Client: search_done / fetch_done
        Orch->>Ollama: tool result + continue
    end

    Ollama-->>Orch: final answer tokens
    Server-->>Client: token (streamed)

    Orch->>Orch: save messages to SQLite
    Server-->>Client: rag_context (if RAG used)
    Server-->>Client: fetch_context (if URLs fetched)
    Server-->>Client: stats (tokens, context%)
    Server-->>Client: done (full content)

    opt First turn in new conversation
        Orch->>Ollama: generate title
        Server-->>Client: title (conv_id, title)
    end

    opt context_pct > COMPRESS_THRESHOLD (70%)
        Orch->>Ollama: summarise old messages
        Orch->>Orch: replace history with summary
        Server-->>Client: compress
    end

    Server->>Server: close SSE connection
```

## Cancel / Stop flow

```mermaid
sequenceDiagram
    participant Client as Client
    participant Server as server.py
    participant Thread as produce() thread

    Client->>Server: POST /cancel
    Server->>Server: cancel_event.set()
    Server-->>Client: 200 OK

    Thread->>Thread: cancel_event.is_set() → break
    Thread->>Server: queue.put(None) sentinel
    Server->>Server: event_stream() receives None
    Server->>Server: truncate conversation_history (rollback)
    Client->>Server: reader.cancel() / close SSE

    note over Client,Server: Turn never saved to DB
```

## RAG Pipeline

```mermaid
flowchart LR
    subgraph ingest["Indexing (on attach)"]
        FILE["Attachment\n(PDF/HTML/text)"] --> FH["file_handler\nextract text"]
        FH --> CHUNK["Chunker\n400 words, 40 overlap"]
        CHUNK --> EMBED["SentenceTransformer\nnomic-embed-text-v1.5\n768 dims\n(local, no Ollama needed)"]
        EMBED --> CHROMA[("ChromaDB\nEphemeralClient\n(in-memory)")]
    end

    subgraph query["Retrieval (each turn)"]
        Q["User message"] --> QE["Embed query\n(SentenceTransformer\nnomic-embed-text-v1.5)"]
        QE --> RET["Cosine similarity\ntop-10 candidates"]
        CHROMA --> RET
        RET --> RERANK["Reranker (default qwen3)\nQwen3-Reranker-0.6B-4bit (mlx, in-process)\nor CrossEncoder ms-marco-MiniLM-L-6-v2"]
        RERANK --> FILTER{"score >\nthreshold?"}
        FILTER -->|yes| INJECT["Inject into\nsystem prompt"]
        FILTER -->|no| DROP["Drop chunk"]
    end

    style ingest fill:#f5f0e8,stroke:#c8b89a
    style query fill:#e8f0f5,stroke:#9ab8c8
```

## iOS Connection Flow

```mermaid
flowchart TD
    START([App Launch]) --> LAST{"Last-used\nURL saved?"}
    LAST -->|yes| PROBE["Probe /health\n(1.4s min splash)"]
    LAST -->|no| CONN_VIEW

    PROBE -->|200 ready| CONNECTED([Connected\niOSConnectedView])
    PROBE -->|fail| FALLBACK["Try saved\nfallback URLs"]
    FALLBACK -->|one responds| CONNECTED
    FALLBACK -->|all fail| CONN_VIEW

    CONN_VIEW["ConnectionView\n(manual URL entry)"]
    CONN_VIEW -->|user enters URL| SAVE["Save URL\nUserDefaults"]
    SAVE --> CONNECTED
```

## macOS Server Startup

```mermaid
flowchart TD
    START([App Launch]) --> LAUNCH["MacConnectionManager\npoll /health every 5s"]
    LAUNCH --> READY{"200 OK\nwithin 60s?"}
    READY -->|yes| ROOT([MacRootView\nConversation sidebar + chat])
    READY -->|no| SPLASH["SplashView\nspinner + status"]
    SPLASH --> RETRY["Retry (up to 5×)\nexponential backoff"]
    RETRY --> READY
    READY -->|timeout| ERROR([Error state\nbackend unreachable])

    subgraph launchd["launchd (login item)"]
        PLIST["com.mab.mira-server.plist\n~/Library/LaunchAgents/"]
        PY["python server.py\n(mira-core)"]
        PLIST --> PY
    end
```
