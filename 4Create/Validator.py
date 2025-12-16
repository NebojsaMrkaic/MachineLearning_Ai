import json
import re
from difflib import SequenceMatcher

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Load error {path}: {e}")
        return {}

def grammar_score(text):
    if not isinstance(text, str) or not text.strip():
        return 0
    penalties = 0
    if not text[0].isupper():
        penalties += 1
    if not re.search(r'[.!?]$', text.strip()):
        penalties += 1
    if re.search(r'\s{2,}', text):
        penalties += 1
    if re.search(r'\s+[,.!?]', text):
        penalties += 1
    return max(0, 5 - penalties)

def flatten_json(obj, prefix=''):
    flat = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat.update(flatten_json(v, f"{prefix}{k}." if prefix else k))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            flat.update(flatten_json(item, f"{prefix}[{i}]."))
    else:
        flat[prefix.rstrip('.')] = obj
    return flat

def compare_fields(json1, json2):
    flat1 = flatten_json(json1)
    flat2 = flatten_json(json2)
    missing_in_2 = set(flat1.keys()) - set(flat2.keys())
    return sorted(missing_in_2)

def compare_grammar(json1, json2):
    flat1 = flatten_json(json1)
    flat2 = flatten_json(json2)
    degraded = []
    for key in flat1:
        if key in flat2 and isinstance(flat1[key], str) and isinstance(flat2[key], str):
            score1 = grammar_score(flat1[key])
            score2 = grammar_score(flat2[key])
            if score2 < score1:
                degraded.append((key, score1, score2))
    return degraded

def similarity_ratio(json1, json2):
    text1 = json.dumps(json1, ensure_ascii=False, indent=2)
    text2 = json.dumps(json2, ensure_ascii=False, indent=2)
    return SequenceMatcher(None, text1, text2).ratio()

def inject_missing_fields(source_json, target_json, missing_keys):
    flat_source = flatten_json(source_json)
    flat_target = flatten_json(target_json)

    for key in missing_keys:
        flat_target[key] = "*** Missing field ***"

    return unflatten_json(flat_target)

def unflatten_json(flat):
    result = {}

    for compound_key, value in flat.items():
        keys = re.split(r'\.(?![^\[]*\])', compound_key)
        current = result

        for i, k in enumerate(keys):
            list_match = re.match(r'(.+?)\[(\d+)\]$', k)
            if list_match:
                key, index = list_match.groups()
                index = int(index)
                if key not in current or not isinstance(current[key], list):
                    current[key] = []
                while len(current[key]) <= index:
                    current[key].append({})
                if i == len(keys) - 1:
                    current[key][index] = value
                else:
                    current = current[key][index]
            else:
                if i == len(keys) - 1:
                    current[k] = value
                else:
                    if k not in current or not isinstance(current[k], dict):
                        current[k] = {}
                    current = current[k]

    return result

def write_report(path, better_file, degraded_file, grammar_issues, missing_fields, similarity):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"✅ Non degraded file: {better_file}\n")
        f.write(f"⚠️ Degradation detected within file: {degraded_file}\n")
        f.write(f"📊 Similarity: {similarity:.2f}\n\n")

        f.write("📝 Grammar differences:\n")
        if grammar_issues:
            for key, s1, s2 in grammar_issues:
                f.write(f"- Field '{key}': score {s1} → {s2}\n")
        else:
            f.write("No grammar issues.\n")

        f.write("\n📉 Missing fields:\n")
        if missing_fields:
            for key in missing_fields:
                f.write(f"- {key}\n")
        else:
            f.write("No missing fields.\n")

        f.write("\n📌 Conclusion:\n")
        f.write(f"File '{better_file}' has been marked as better due to more information, better structure, and grammar.\n")
        f.write(f"File '{degraded_file}' shows degradation compared to the other file.\n")

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    file1 = "output_v1.json"
    file2 = "output_v2.json"

    json1 = load_json(file1)
    json2 = load_json(file2)

    grammar_issues = compare_grammar(json1, json2)
    missing_fields = compare_fields(json1, json2)
    similarity = similarity_ratio(json1, json2)

    if grammar_issues or missing_fields:
        better_file = file1
        degraded_file = file2
        degraded_json = inject_missing_fields(json1, json2, missing_fields)
    else:
        better_file = file2
        degraded_file = file1
        missing_fields = compare_fields(json2, json1)
        degraded_json = inject_missing_fields(json2, json1, missing_fields)

    write_report("differences.txt", better_file, degraded_file, grammar_issues, missing_fields, similarity)
    save_json("degraded_with_missing.json", degraded_json)
    print("✅ differences.txt and degraded_with_missing.json have been generated.")

if __name__ == "__main__":
    main()
