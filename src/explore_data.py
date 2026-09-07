from pathlib import Path
import pandas as pd
from PIL import Image

data_dir = Path("data/pokemon_images/sprites")
png_files = data_dir.rglob("*.png")
filtered_list=[]

for path in (png_files):
    if path.parts[4] == 'front':
        filtered_list.append(path)

print("Number of images considering only front images of pokemons: ",len(filtered_list))
print("Sanity check of first image path: ",filtered_list[0])

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

print("Number of dictionaries for each file with 'path', 'species', 'dex', and 'variant' key",len(records))
print("First pokemon in the dictionary: ", records[0])

species_set = set()
for record in records:
    species_set.add(record["species"])

print("Total number of unique species: ", len(species_set))


csv_pat = Path("data/pokemon_images/pokedex.csv")
df = pd.read_csv(csv_pat)
print("Pokedex.csv shape: ", df.shape)
print("First 5 rows of pokedex.csv:")
print(df.head())

pandas_name_set = set(df["name"])
print("Total of species in pokedex.csv: ", len(pandas_name_set))

in_csv_not_images = pandas_name_set - species_set
in_images_not_csv = species_set - pandas_name_set
print("Species in csv but not in images set: ", len(in_csv_not_images))
print("Species in images set but not in csv: ", len(in_images_not_csv))


image_sizes = set()
image_modes = set()
corrupted = []

print("Number of images prior checking mode and size of images: ", len(records))

for record in records:
    try: 
        img = Image.open(record["path"])
        img.load()
        image_sizes.add(img.size)
        image_modes.add(img.mode)
        
    except Exception as e:
        corrupted.append((record["path"], str(e)))

print("image sizes: ", image_sizes)
print("image modes: ", image_modes)
print("Total of corrupted files: ", len(corrupted))