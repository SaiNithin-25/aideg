import argparse
import json
from pathlib import Path


DEFAULT_JSONL_PATH = Path(__file__).resolve().parents[1] / "work" / "video_0" / "transcript.jsonl"


def count_jsonl_samples(file_path: Path, validate_json: bool = False) -> tuple[int, int]:
    total_lines = 0
    valid_samples = 0

    with file_path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue

            total_lines += 1

            if validate_json:
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Invalid JSON at line {line_number}: {exc}")
                    continue

            valid_samples += 1

    return total_lines, valid_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Count samples in a JSONL file.")
    parser.add_argument(
        "file_path",
        nargs="?",
        default=str(DEFAULT_JSONL_PATH),
        help="Path to the .jsonl file. Defaults to work/video_0/transcript.jsonl",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Parse each non-empty line as JSON before counting it as a valid sample.",
    )
    args = parser.parse_args()

    file_path = Path(args.file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    total_lines, valid_samples = count_jsonl_samples(file_path, validate_json=args.validate)

    print(f"File: {file_path}")
    print(f"Samples/lines: {total_lines}")

    if args.validate:
        print(f"Valid JSON samples: {valid_samples}")


if __name__ == "__main__":
    main()
