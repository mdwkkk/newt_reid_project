"""Шаблоны карточек особей (ИК-1, ИК-2, КВ-1, КВ-2)."""
from __future__ import annotations

from typing import Any, Literal

CardTemplateCode = Literal["IK-1", "IK-2", "KV-1", "KV-2"]

FieldType = Literal[
    "text",
    "date",
    "time",
    "number",
    "select",
    "textarea",
    "individual_ref",
]

TEMPLATE_CODES: list[CardTemplateCode] = ["IK-1", "IK-2", "KV-1", "KV-2"]


def _field(
    key: str,
    label_ru: str,
    field_type: FieldType = "text",
    *,
    required: bool = False,
    options: list[str] | None = None,
) -> dict[str, Any]:
    f: dict[str, Any] = {
        "key": key,
        "label_ru": label_ru,
        "type": field_type,
        "required": required,
    }
    if options is not None:
        f["options"] = options
    return f


CARD_TEMPLATES: dict[str, dict[str, Any]] = {
    "IK-1": {
        "code": "IK-1",
        "name_ru": "ИК-1",
        "description": "Индивидуальная карточка (лабораторный учёт)",
        "fields": [
            _field("individual_id", "ID-номер особи", required=True),
            _field("card_date", "Дата заполнения карточки", "date"),
            _field("body_length", "Длина тела (L)", "number"),
            _field("tail_length", "Длина хвоста (Lcd)", "number"),
            _field("mass_g", "Масса, г.", "number"),
            _field("sex", "Пол", "select", options=["самец", "самка", "неизвестно"]),
            _field("birth_date_exact", "Год рождения особи точный (дд.мм.гггг)", "date"),
            _field("birth_date_approx", "Условный год рождения особи (дд.мм.гггг)", "date"),
            _field("pattern_photo_no", "Номер фото индивидуального рисунка"),
            _field("origin_region", "Регион происхождения особи"),
            _field("length_device", "Марка устройства, использованного для измерения длины"),
            _field("scale_brand", "Марка весов, использованных для взвешивания особи"),
            _field("notes", "Примечания", "textarea"),
        ],
    },
    "IK-2": {
        "code": "IK-2",
        "name_ru": "ИК-2",
        "description": "Индивидуальная карточка (разведение)",
        "fields": [
            _field("individual_id", "ID-номер особи", required=True),
            _field("card_date", "Дата заполнения карточки", "date"),
            _field("release_date", "Дата выпуска в водоём", "date"),
            _field("parent_male_id", "ID самца (родитель)", "individual_ref"),
            _field("parent_female_id", "ID самки (родитель)", "individual_ref"),
            _field("total_length_cm", "Общая длина (L + Lcd), см", "number"),
            _field("mass_g", "Масса, г.", "number"),
            _field("pond_name", "Название водоёма"),
            _field("notes", "Примечания", "textarea"),
        ],
    },
    "KV-1": {
        "code": "KV-1",
        "name_ru": "КВ-1",
        "description": "Карточка встречи (подробная)",
        "fields": [
            _field("individual_id", "ID-номер особи", required=True),
            _field("encounter_date", "Дата встречи (дд.мм.гггг)", "date"),
            _field("encounter_time", "Время встречи (ч. мин.)", "time"),
            _field("body_length_mm", "Длина тела (L), мм", "number"),
            _field("tail_length_mm", "Длина хвоста (Lcd), мм", "number"),
            _field("mass_g", "Масса, г.", "number"),
            _field("sex", "Пол", "select", options=["самец", "самка", "неизвестно"]),
            _field("belly_pattern_photo_no", "Номер фото рисунка брюшной стороны"),
            _field("status", "Статус", "select", options=["жив", "мертв"]),
            _field("pond_number", "Номер водоёма"),
            _field("length_device", "Марка устройства для измерения длины"),
            _field("scale_brand", "Марка весов"),
            _field("notes", "Примечания", "textarea"),
        ],
    },
    "KV-2": {
        "code": "KV-2",
        "name_ru": "КВ-2",
        "description": "Карточка встречи (краткая)",
        "fields": [
            _field("individual_id", "ID-номер особи", required=True),
            _field("encounter_date", "Дата встречи (дд.мм.гггг)", "date"),
            _field("encounter_time", "Время встречи (ч. мин.)", "time"),
            _field("total_length_cm", "Общая длина (L + Lcd), см", "number"),
            _field("status", "Статус", "select", options=["жив", "мертв"]),
            _field("pond_name", "Название водоёма"),
            _field("notes", "Примечания", "textarea"),
        ],
    },
}


def list_templates() -> list[dict[str, Any]]:
    return [
        {
            "code": t["code"],
            "name_ru": t["name_ru"],
            "description": t.get("description", ""),
            "field_count": len(t["fields"]),
        }
        for t in CARD_TEMPLATES.values()
    ]


def get_template(code: str) -> dict[str, Any] | None:
    return CARD_TEMPLATES.get(code)


def is_valid_template(code: str) -> bool:
    return code in CARD_TEMPLATES


def validate_metadata(
    card_template: str,
    metadata: dict[str, Any],
    *,
    existing_ids: set[str] | None = None,
    exclude_individual_id: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Валидация и нормализация metadata. Возвращает (cleaned, errors)."""
    tpl = get_template(card_template)
    if tpl is None:
        return {}, [f"Неизвестный шаблон: {card_template}"]

    errors: list[str] = []
    cleaned: dict[str, Any] = {}

    individual_id = str(metadata.get("individual_id", "")).strip()
    if not individual_id:
        errors.append("ID-номер особи обязателен")
    else:
        cleaned["individual_id"] = individual_id
        if existing_ids is not None:
            if exclude_individual_id and individual_id == exclude_individual_id:
                pass
            elif individual_id in existing_ids:
                errors.append(f"Особь с ID «{individual_id}» уже существует в проекте")

    for field in tpl["fields"]:
        key = field["key"]
        if key == "individual_id":
            continue
        raw = metadata.get(key)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            cleaned[key] = ""
            if field.get("required"):
                errors.append(f"Поле «{field['label_ru']}» обязательно")
            continue
        val = raw if not isinstance(raw, str) else raw.strip()
        cleaned[key] = val

        ftype = field["type"]
        if ftype == "individual_ref" and val:
            if existing_ids is not None and val not in existing_ids:
                if val != exclude_individual_id:
                    errors.append(
                        f"«{field['label_ru']}»: особь «{val}» не найдена в проекте"
                    )
        if ftype == "select" and val:
            opts = field.get("options") or []
            if val not in opts:
                errors.append(
                    f"«{field['label_ru']}»: недопустимое значение «{val}»"
                )

    return cleaned, errors
