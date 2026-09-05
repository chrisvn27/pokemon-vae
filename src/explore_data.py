from pathlib import Path
import pandas as pd


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

species_set = set()

for record in records:
    species_set.add(record["species"])

print(len(species_set))
# print(species_set)

csv_pat = Path("data/pokemon_images/pokedex.csv")
df = pd.read_csv(csv_pat)
print(df.shape)
print(df.head())

pandas_name_set = set(df["name"])
print(len(pandas_name_set))

in_csv_not_images = pandas_name_set - species_set
in_images_not_csv = species_set - pandas_name_set
print(len(in_csv_not_images))
print(len(in_images_not_csv))
print(in_csv_not_images)
print(in_images_not_csv)

id= df[df["name"] == "Pikachu Phd"]["id"]
print(id)