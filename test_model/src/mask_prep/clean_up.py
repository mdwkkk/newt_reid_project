import os
import shutil

masks_dir = 'data/masks'

print("🔍 Поиск файлов для удаления...")
print("=" * 60)

json_files = [f for f in os.listdir(masks_dir) if f.endswith('.json')]
json_folders = [f for f in os.listdir(masks_dir) if f.endswith('_json')]

print(f"📄 JSON файлов: {len(json_files)}")
print(f"📂 Папок _json: {len(json_folders)}")
print("=" * 60)

if len(json_files) == 0 and len(json_folders) == 0:
    print("✅ Нечего удалять!")
    exit()

print("\n📋 Будет удалено:")
print("-" * 60)

for f in json_files:
    print(f"  📄 {f}")

for f in json_folders:
    print(f"  📂 {f}/")

print("-" * 60)

response = input("\n⚠️ Удалить? (y/n): ")

if response.lower() != 'y':
    print("❌ Отменено.")
    exit()

print("\n🗑️ Удаление...")

for filename in json_files:
    file_path = os.path.join(masks_dir, filename)
    try:
        os.remove(file_path)
        print(f"  ✅ {filename}")
    except Exception as e:
        print(f"  ❌ {filename} - {e}")

for folder in json_folders:
    folder_path = os.path.join(masks_dir, folder)
    try:
        shutil.rmtree(folder_path)
        print(f"  ✅ {folder}/")
    except Exception as e:
        print(f"  ❌ {folder}/ - {e}")

print("=" * 60)
print("🎉 Очистка завершена!")