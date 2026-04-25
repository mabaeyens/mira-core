"""Tool definitions for Ollama API."""

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information. Use this when the user asks about recent events, news, prices, or anything after April 2024.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}

FETCH_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": (
            "Fetch the full text content of a web page. "
            "Use this when a web_search result looks relevant but the snippet "
            "is too short to answer the question — fetch the page to read the details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch"
                }
            },
            "required": ["url"]
        }
    }
}


# ── Filesystem tools ──────────────────────────────────────────────────────────

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file in the workspace. Path is relative to the workspace root.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path within the workspace"},
            },
            "required": ["path"],
        },
    },
}

WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Create or overwrite a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path"},
                "content": {"type": "string", "description": "Full file content to write"},
            },
            "required": ["path", "content"],
        },
    },
}

LIST_FILES_TOOL = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files and directories in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: workspace root)", "default": "."},
                "recursive": {"type": "boolean", "description": "Include subdirectories recursively", "default": False},
            },
        },
    },
}

SEARCH_FILES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "Search file contents with a regex pattern (grep-like). Returns matching lines with file and line number.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory to search in (default: workspace root)", "default": "."},
                "case_sensitive": {"type": "boolean", "default": False},
            },
            "required": ["pattern"],
        },
    },
}

EDIT_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": (
            "Replace an exact string in a file. "
            "old_str must match exactly once — if it matches zero or multiple times the edit is rejected. "
            "Prefer this over write_file for targeted changes to existing files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path within the workspace"},
                "old_str": {"type": "string", "description": "Exact string to replace (must be unique in the file)"},
                "new_str": {"type": "string", "description": "Replacement string"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}

MOVE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "move_file",
        "description": "Move or rename a file within the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "src": {"type": "string", "description": "Source path"},
                "dst": {"type": "string", "description": "Destination path"},
            },
            "required": ["src", "dst"],
        },
    },
}

DELETE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "delete_file",
        "description": "Delete a file or directory from the workspace. Requires confirm=true after user approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "confirm": {"type": "boolean", "description": "Must be true to execute; omit to get a confirmation prompt first", "default": False},
            },
            "required": ["path"],
        },
    },
}

# ── Shell tool ────────────────────────────────────────────────────────────────

RUN_SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": "Run a shell command. Working directory is within the workspace. Destructive commands (rm -rf, git reset --hard, etc.) require force=true after user approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "cwd": {"type": "string", "description": "Working directory relative to workspace root (default: root)", "default": "."},
                "force": {"type": "boolean", "description": "Set true to run a previously flagged dangerous command after user confirms", "default": False},
            },
            "required": ["command"],
        },
    },
}

# ── GitHub tools ──────────────────────────────────────────────────────────────

GITHUB_CLONE_REPO_TOOL = {
    "type": "function",
    "function": {
        "name": "github_clone_repo",
        "description": (
            "Clone a GitHub repository to a local path and register it as a Mira project. "
            "Use this when the user wants to work with a GitHub repo locally."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/repo"},
                "dest": {"type": "string", "description": "Local destination path (default: ~/workspace/<repo-name>)", "default": ""},
                "project_name": {"type": "string", "description": "Name for the Mira project (default: repo name)", "default": ""},
            },
            "required": ["repo"],
        },
    },
}

GITHUB_LIST_REPOS_TOOL = {
    "type": "function",
    "function": {
        "name": "github_list_repos",
        "description": "List the authenticated user's GitHub repositories.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo_type": {"type": "string", "description": "owner, member, or all", "default": "owner"},
            },
        },
    },
}

GITHUB_READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "github_read_file",
        "description": "Read a file's content from a GitHub repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/repo"},
                "path": {"type": "string", "description": "File path in repo"},
                "ref": {"type": "string", "description": "Branch, tag, or commit SHA (default: default branch)", "default": ""},
            },
            "required": ["repo", "path"],
        },
    },
}

GITHUB_LIST_FILES_TOOL = {
    "type": "function",
    "function": {
        "name": "github_list_files",
        "description": "List files and directories at a path in a GitHub repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/repo"},
                "path": {"type": "string", "description": "Directory path (default: repo root)", "default": ""},
                "ref": {"type": "string", "description": "Branch, tag, or commit SHA", "default": ""},
            },
            "required": ["repo"],
        },
    },
}

GITHUB_WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "github_write_file",
        "description": "Create or update a file in a GitHub repository (auto-commits).",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/repo"},
                "path": {"type": "string", "description": "File path in repo"},
                "content": {"type": "string", "description": "Full file content"},
                "message": {"type": "string", "description": "Commit message"},
                "branch": {"type": "string", "description": "Target branch (default: default branch)", "default": ""},
            },
            "required": ["repo", "path", "content", "message"],
        },
    },
}

GITHUB_CREATE_REPO_TOOL = {
    "type": "function",
    "function": {
        "name": "github_create_repo",
        "description": "Create a new GitHub repository under the authenticated user's account.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Repository name"},
                "private": {"type": "boolean", "description": "Create as private (default: true)", "default": True},
                "description": {"type": "string", "description": "Repository description", "default": ""},
                "auto_init": {"type": "boolean", "description": "Initialize with a README", "default": True},
            },
            "required": ["name"],
        },
    },
}

GITHUB_CREATE_BRANCH_TOOL = {
    "type": "function",
    "function": {
        "name": "github_create_branch",
        "description": "Create a new branch in a GitHub repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/repo"},
                "branch": {"type": "string", "description": "New branch name"},
                "from_ref": {"type": "string", "description": "Source branch/commit (default: default branch)", "default": ""},
            },
            "required": ["repo", "branch"],
        },
    },
}

GITHUB_LIST_ISSUES_TOOL = {
    "type": "function",
    "function": {
        "name": "github_list_issues",
        "description": "List issues in a GitHub repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/repo"},
                "state": {"type": "string", "description": "open, closed, or all", "default": "open"},
            },
            "required": ["repo"],
        },
    },
}

GITHUB_CREATE_ISSUE_TOOL = {
    "type": "function",
    "function": {
        "name": "github_create_issue",
        "description": "Create a new issue in a GitHub repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/repo"},
                "title": {"type": "string"},
                "body": {"type": "string", "description": "Issue description in Markdown", "default": ""},
            },
            "required": ["repo", "title"],
        },
    },
}

GITHUB_LIST_PRS_TOOL = {
    "type": "function",
    "function": {
        "name": "github_list_prs",
        "description": "List pull requests in a GitHub repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/repo"},
                "state": {"type": "string", "description": "open, closed, or all", "default": "open"},
            },
            "required": ["repo"],
        },
    },
}

GITHUB_SEARCH_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "github_search_code",
        "description": "Search code on GitHub.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (GitHub code search syntax)"},
                "repo": {"type": "string", "description": "Restrict search to owner/repo (optional)", "default": ""},
            },
            "required": ["query"],
        },
    },
}

GITHUB_CREATE_PR_TOOL = {
    "type": "function",
    "function": {
        "name": "github_create_pr",
        "description": "Open a pull request on GitHub. head is the branch with your changes; base is where to merge (defaults to the repo's default branch).",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/repo"},
                "title": {"type": "string"},
                "body": {"type": "string", "description": "PR description in Markdown", "default": ""},
                "head": {"type": "string", "description": "Branch that contains the changes"},
                "base": {"type": "string", "description": "Branch to merge into (default: repo default branch)", "default": ""},
            },
            "required": ["repo", "title", "head"],
        },
    },
}

GITHUB_MERGE_PR_TOOL = {
    "type": "function",
    "function": {
        "name": "github_merge_pr",
        "description": "Merge a pull request. Requires confirm=true after user approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/repo"},
                "pr_number": {"type": "integer", "description": "PR number"},
                "merge_method": {"type": "string", "description": "merge, squash, or rebase", "default": "merge"},
                "confirm": {"type": "boolean", "default": False},
            },
            "required": ["repo", "pr_number"],
        },
    },
}

GITHUB_DELETE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "github_delete_file",
        "description": "Delete a file from a GitHub repository. Requires confirm=true after user approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/repo"},
                "path": {"type": "string", "description": "File path in repo"},
                "message": {"type": "string", "description": "Commit message"},
                "branch": {"type": "string", "description": "Branch (default: default branch)", "default": ""},
                "confirm": {"type": "boolean", "default": False},
            },
            "required": ["repo", "path", "message"],
        },
    },
}

GITHUB_DELETE_BRANCH_TOOL = {
    "type": "function",
    "function": {
        "name": "github_delete_branch",
        "description": "Delete a branch from a GitHub repository. Requires confirm=true after user approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/repo"},
                "branch": {"type": "string"},
                "confirm": {"type": "boolean", "default": False},
            },
            "required": ["repo", "branch"],
        },
    },
}

TOOLS = [
    SEARCH_TOOL, FETCH_TOOL,
    # Filesystem
    READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL, LIST_FILES_TOOL, SEARCH_FILES_TOOL,
    MOVE_FILE_TOOL, DELETE_FILE_TOOL,
    # Shell
    RUN_SHELL_TOOL,
    # GitHub
    GITHUB_CLONE_REPO_TOOL,
    GITHUB_LIST_REPOS_TOOL, GITHUB_READ_FILE_TOOL, GITHUB_LIST_FILES_TOOL,
    GITHUB_WRITE_FILE_TOOL, GITHUB_CREATE_REPO_TOOL, GITHUB_CREATE_BRANCH_TOOL,
    GITHUB_LIST_ISSUES_TOOL, GITHUB_CREATE_ISSUE_TOOL, GITHUB_LIST_PRS_TOOL,
    GITHUB_SEARCH_CODE_TOOL, GITHUB_CREATE_PR_TOOL, GITHUB_MERGE_PR_TOOL,
    GITHUB_DELETE_FILE_TOOL, GITHUB_DELETE_BRANCH_TOOL,
]

# Tool names that require a local workspace — excluded when project has no local_path
_LOCAL_TOOLS = {
    "read_file", "write_file", "edit_file", "list_files", "search_files",
    "move_file", "delete_file", "run_shell",
}