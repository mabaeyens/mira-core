# Homebrew formula for Mira. Canonical source lives here in the mira-core repo;
# copy it into the tap (mabaeyens/homebrew-mira) on release. See README.md in this
# directory for the tap setup, the release-bump step, and the design tradeoffs.
class Mira < Formula
  desc "Local AI assistant — FastAPI backend, RAG, web UI (Apple Silicon)"
  homepage "https://github.com/mabaeyens/mira-core"
  url "https://github.com/mabaeyens/mira-core/archive/refs/tags/v1.4.0.tar.gz"
  sha256 "b2167bd1d56482560abac76e33bdb19c45fd34de5393af6c5d48a5256724c9a2"
  license "MIT"

  # uv owns the Python side (venv + the large MLX/ML dependency tree). Enumerating
  # every wheel as a Homebrew `resource` would be enormous and fragile, so the
  # formula installs the source and lets uv build the env on first run instead.
  depends_on "uv"
  depends_on macos: :ventura # MLX needs a recent macOS; Apple Silicon enforced by setup.sh

  def install
    libexec.install Dir["*"]

    # Mira lives in the Cellar (read-mostly, wiped on upgrade), so the launcher
    # keeps every mutable path in $HOME: config via MIRA_CONFIG, and data
    # (conversations.db, chroma) already defaults to ~/.local/share/mira.
    (bin/"mira").write <<~SH
      #!/bin/bash
      export MIRA_HOME="#{libexec}"
      export MIRA_CONFIG="${MIRA_CONFIG:-$HOME/.config/mira/mira.yaml}"
      cd "#{libexec}" || exit 1

      # doctor / preflight / --help are stdlib-only — run them on the system
      # python so they work the instant `brew install` finishes, before the
      # one-time dependency sync.
      case "${1:-}" in
        doctor|preflight|-h|--help|"")
          exec /usr/bin/python3 "#{libexec}/mira_cli.py" "$@" ;;
      esac

      # Everything else needs the venv. Homebrew's build sandbox blocks network,
      # so deps can't sync at install time; `uv run` creates/syncs the env on
      # first use here instead (one-time, ~1–2 min, with its own progress).
      exec uv run --project "#{libexec}" python "#{libexec}/mira_cli.py" "$@"
    SH
    chmod 0755, bin/"mira"
  end

  def caveats
    <<~EOS
      Mira runs local inference on Apple Silicon and needs a model on disk.

      First run, in order:
        mira setup          # seed ~/.config/mira/mira.yaml, check prerequisites
        mira fetch-model    # one-time ~19 GB default-model download (mira-mlx)
        mira serve          # start the web server on http://localhost:8000

      Config:  ~/.config/mira/mira.yaml
      Data:    ~/.local/share/mira   (conversations, RAG index)
      Both survive `brew upgrade`; the bundled venv is rebuilt on first run after
      an upgrade.

      The default backend (mira-mlx) needs no extra app. To use oMLX instead,
      install oMLX.app from https://github.com/jundot/omlx/releases and set
      `backend: omlx` in the config above — `mira doctor` will guide you.
    EOS
  end

  test do
    assert_match "usage: mira", shell_output("#{bin}/mira --help")
  end
end
