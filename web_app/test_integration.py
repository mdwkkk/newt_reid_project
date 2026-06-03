"""Интеграционные проверки без весов модели (store + templates + DB)."""
from web_app.card_templates import CARD_TEMPLATES, list_templates, validate_metadata
from web_app.services import store

store.init_db()

assert len(list_templates()) == 4
for code in CARD_TEMPLATES:
    assert code in ("IK-1", "IK-2", "KV-1", "KV-2")

p = store.create_project("QA Empty", "IK-1")
pid = p["id"]
assert store.count_individuals(pid) == 0

meta, errs = validate_metadata("IK-1", {"individual_id": "qa001"}, existing_ids=set())
assert not errs
store.create_individual(pid, "qa001", meta)
assert store.count_individuals(pid) == 1

ind = store.get_individual(pid, "qa001")
assert ind is not None

meta2, errs2 = validate_metadata(
    "IK-2",
    {"individual_id": "qa002", "parent_male_id": "qa001"},
    existing_ids={"qa001"},
)
assert not errs2

store.delete_project(pid)
assert store.get_project(pid) is None

print("integration OK")
