import json
from pathlib import Path
from tqdm import tqdm

# gdown 1nJ86OLnF2C6eDoi5UOAdTAS5Duc0wuTl
# gdown 1OXlNBuW74dsrwYZIpQMshFqxkjcMPPgV
# mv pid_splits.json scienceqa_pid_splits.json
# mv problems.json scienceqa_problems_path.json
# mkdir images && cd images
# gdown 1swX4Eei1ZqrXRvM-JAZxN6QVwcBLPHV8
# gdown 1eyjFaHxbvEJZzdZILn3vnTihBNDmKcIj
# gdown 1ijThWZc1tsoqGrOCWhYYj1HUJ48Hl8Zz

def convert(dataset_dir, data, ids, split):
    annotation = []
    for id in tqdm(ids):
        d = data[str(id)]
        if d['image'] is None:
            continue
        answer = d['answer']
        letter = 'abcde'[min(answer, 4)]
        annotation.append({
            "image": f"images/{split}/{id}/image.png",
            "question": d['question'],
            "answer" : f"({letter}) {d['choices'][answer]}",
            "choices": d['choices'],
            "context" : f"{d['hint']} {d['lecture']}",
            "question_id" : id,
        })

    out_fname = dataset_dir / f"scienceqa_{split}.json"
    with open(out_fname, 'w') as file:
        json.dump(annotation, file)
    print(out_fname)
    return out_fname, annotation


def main(dataset_dir):
    dataset_dir = Path(dataset_dir).resolve()
    with open(dataset_dir / "scienceqa_problems_path.json", 'r') as file:
        data = json.load(file)

    with open(dataset_dir / "scienceqa_pid_splits.json") as file:
        pid_splits = json.load(file)

    # make train annotation
    convert(dataset_dir, data, pid_splits['train'], 'train')
    convert(dataset_dir, data, pid_splits['val'], 'val')
    convert(dataset_dir, data, pid_splits['test'], 'test')

if __name__ == '__main__':
    import fire
    fire.Fire(main)