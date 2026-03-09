import argparse
import sys
from pathlib import Path

from flask import Flask, render_template, request

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sentiment_regression.prediction import predict_single_text

def create_app(model_dir: Path) -> Flask:
    app = Flask(__name__)
    app.config["MODEL_DIR"] = str(model_dir)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            text="",
            result=None,
            error=None,
            model_dir=str(model_dir),
        )

    @app.post("/")
    def score():
        text = (request.form.get("text") or "").strip()
        if not text:
            return render_template(
                "index.html",
                text="",
                result=None,
                error="Please enter a sentence.",
                model_dir=str(model_dir),
            )

        try:
            result = predict_single_text(model_dir=model_dir, text=text)
        except Exception as exc:  # keep UI friendly
            return render_template(
                "index.html",
                text=text,
                result=None,
                error=str(exc),
                model_dir=str(model_dir),
            )

        return render_template(
            "index.html",
            text=text,
            result=result,
            error=None,
            model_dir=str(model_dir),
        )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Local web UI for sentiment regression.")

    default_model_dir = "models/sentiment140_run"
    if not (Path(default_model_dir) / "model.joblib").exists():
        default_model_dir = "models/sample_run"

    parser.add_argument(
        "--model-dir",
        default=default_model_dir,
        help="Folder with model.joblib + metadata.json (default: auto).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000).")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    app = create_app(model_dir=model_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
