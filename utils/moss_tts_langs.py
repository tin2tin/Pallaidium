# Language map for MOSS-TTS (OpenMOSS-Team/MOSS-TTS family).
# Maps language code -> display name for the Blender EnumProperty.
# MOSS-TTS-v1.5 officially supports the 31 codes below.

_LANG_DATA = [
    ("zh",  "Chinese"),
    ("yue", "Cantonese"),
    ("en",  "English"),
    ("ar",  "Arabic"),
    ("cs",  "Czech"),
    ("da",  "Danish"),
    ("nl",  "Dutch"),
    ("fi",  "Finnish"),
    ("fr",  "French"),
    ("de",  "German"),
    ("el",  "Greek"),
    ("he",  "Hebrew"),
    ("hi",  "Hindi"),
    ("hu",  "Hungarian"),
    ("it",  "Italian"),
    ("ja",  "Japanese"),
    ("ko",  "Korean"),
    ("mk",  "Macedonian"),
    ("ms",  "Malay"),
    ("fa",  "Persian"),
    ("pl",  "Polish"),
    ("pt",  "Portuguese"),
    ("ro",  "Romanian"),
    ("ru",  "Russian"),
    ("es",  "Spanish"),
    ("sw",  "Swahili"),
    ("sv",  "Swedish"),
    ("tl",  "Tagalog"),
    ("th",  "Thai"),
    ("tr",  "Turkish"),
    ("vi",  "Vietnamese"),
]

# Sorted alphabetically by display name, with Auto first.
MOSS_LANG_ITEMS = [
    ("AUTO", "Auto", "Let MOSS-TTS infer the language from the prompt text"),
]
MOSS_LANG_ITEMS += [
    (code, name, f"Language code: {code}")
    for code, name in sorted(_LANG_DATA, key=lambda x: x[1])
]
