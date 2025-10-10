"""
Helper függvények UI komponensek frissítéséhez.
Jelenleg nem használt, legacy kód a korábbi UI verzióból deee azért jó lenne valahogy használatba hozni
"""


def update_result(class_name, confidence):
    class_text = f"🎯 {class_name.upper()}" if class_name != "-" else "🎯 -"
    conf_text = f"📊 {confidence:.1f}%" if confidence != "-" else "📊 -%"
    return class_text, conf_text


def update_debug_log(message, existing_log=""):
    if existing_log:
        return f"{existing_log}\n{message}"
    return message


def format_pipeline_info(preprocessing_name):
    return f"Preprocessing: {preprocessing_name}"