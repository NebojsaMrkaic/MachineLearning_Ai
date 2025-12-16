import argparse
import json
from pathlib import Path
from datetime import datetime

class DataSource:
    def __init__(self, path: Path):
        self.path = Path(path)

    def read(self):
        if not self.path.exists():
            print(f" Input file not found: {self.path}, skipping.")
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f" Skipping invalid JSON line in {self.path}: {e}")
        return records


class Validator:
    def __init__(self, rules):
        self.rules = rules

    def apply(self, records):
        ok, ko = [], []
        for rec in records:
            errors = {}
            for rule in self.rules:
                field = rule["field"]
                for r in rule["validations"]:
                    if not self._check(rec, field, r):
                        errors.setdefault(field, []).append(r)
            if errors:
                rec["validation_errors"] = errors
                ko.append(rec)
            else:
                ok.append(rec)
        return ok, ko

    def _check(self, rec, field, rule):
        val = rec.get(field)
        if rule == "notNull":
            return val is not None
        if rule == "notEmpty":
            return isinstance(val, str) and val.strip() != ""
        if rule == "isInt":
            return isinstance(val, int)
        if rule.startswith("range:"):
            _, min_v, max_v = rule.split(":")
            return isinstance(val, int) and int(min_v) <= val <= int(max_v)
        return True


class Transformer:
    @staticmethod
    def add_ingestion_date(records):
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        for rec in records:
            rec["ingestion_dt"] = now
        return records

class Sink:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, records):
        with self.path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"✅ Wrote {len(records)} records to {self.path}")

def run_transformation(step, context):
    name = step.get("name")
    ttype = step.get("type")
    params = step.get("params", {})

    if ttype == "validate_fields":
        input_names = params.get("input")
        if isinstance(input_names, str):
            input_names = [input_names]
        records = []
        for in_name in input_names:
            records.extend(context.get(in_name, []))
        validations = params.get("validations", [])
        ok, ko = Validator(validations).apply(records)
        context[f"{name}_ok"] = ok
        context[f"{name}_ko"] = ko
        context["validation_ok"] = ok
        context["validation_ko"] = ko
        return context

    if ttype == "add_fields":
        input_name = params.get("input")
        records = list(context.get(input_name, []))
        add_fields = params.get("addFields", [])
        want_ingestion = any(
            f.get("function") == "current_timestamp" and f.get("name") == "ingestion_dt"
            for f in add_fields
        )
        if want_ingestion:
            records = Transformer.add_ingestion_date(records)
        context[name] = records
        context["add_ingestion_date"] = records
        return context

    print(f" Unknown transformation type '{ttype}' in step '{name}', skipping.")
    return context


class Pipeline:
    def __init__(self, metadata):
        self.metadata = metadata

    def run(self):
        for flow in self.metadata["dataflows"]:
            print(f"▶️ Running flow: {flow.get('name', 'unnamed')}")
            context = {}

            for src in flow.get("sources", []):
                src_name = src.get("name")
                src_path = Path(src.get("path"))
                data = DataSource(src_path).read()
                if not src_name:
                    print(f" Source without name for path {src_path}, skipping.")
                    continue
                context[src_name] = data

            for step in flow.get("transformations", []):
                context = run_transformation(step, context)

            for sink in flow.get("sinks", []):
                input_key = sink.get("input")
                paths = sink.get("paths", [])
                if not paths:
                    print(" Sink without paths, skipping.")
                    continue

                sink_dir = Path(paths[0])

                if input_key in ("validation_ko", "validationKO", "ko"):
                    ds = context.get("validation_ko") or context.get("validate_fields_ko")
                    Sink(sink_dir / "ko.jsonl").write(ds or [])
                elif input_key in ("validation_ok", "validationOK", "ok"):
                    ds = context.get("validation_ok") or context.get("validate_fields_ok")
                    Sink(sink_dir / "ok.jsonl").write(ds or [])
                else:
                    ds = context.get(input_key)
                    if ds is not None:
                        Sink(sink_dir / "ok.jsonl").write(ds)
                    else:
                        print(f" Sink input '{input_key}' not found. Available: {list(context.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motor policy pipeline")
    parser.add_argument("--metadata", default="data/input/motor_policy.json",
                        help="Path to metadata JSON file")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    Pipeline(metadata).run()
    print("🏁 Pipeline finished. Check data/output for results.")

    #docker compose up --build

