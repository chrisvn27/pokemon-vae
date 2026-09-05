from pathlib import Path

data_dir = Path("data/pokemon_images/sprites")

png_files = data_dir.rglob("*.png")

filtered_list=[]

for path in (png_files):
    if path.parts[4] == 'front':
        filtered_list.append(path)

print(len(filtered_list))
print(filtered_list[0])

records = []

for path in filtered_list:
    parts = path.parts[3]
    parts_1 = parts.split('-',1)
    parts_2 = parts_1[1].rsplit('-',1)
    parts = [parts_1[0]] + parts_2
    record = {
        "path": path,
        "species": parts[1],
        "dex": parts[-1],
        "variant": path.parts[5]
    }
    records.append(record)

print(len(records))
print(records[0])