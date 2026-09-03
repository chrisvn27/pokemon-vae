from pathlib import Path

data_dir = Path("data/pokemon_images/sprites")

png_files = data_dir.rglob("*.png")

filtered_list=[]

for path in (png_files):
    if path.parts[4] == 'front':
        filtered_list.append(path.parts)

print(len(filtered_list))
print(filtered_list[0])

