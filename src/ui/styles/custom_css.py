"""
Custom CSS styles a Gradio interface-hez.
"""


def get_custom_css():
    return """
    .gradio-container {
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
        max-width: 1900px !important;
    }

    .gradio-group {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }

    div.svelte-1nguped {
        background: transparent !important;
    }

    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .app-header h1 {
        margin: 0 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    .app-header p {
        margin: 0.5rem 0 0 0 !important;
        opacity: 0.95;
    }

    .pipeline-step-small {
        background: var(--background-fill-primary);
        padding: 1.25rem;
        border-radius: 12px;
        border: 2px solid var(--border-color-primary);
        min-width: 180px;
        transition: all 0.3s;
    }

    .pipeline-step-small.active {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    }

    .pipeline-step-small.disabled {
        opacity: 0.5;
    }

    .pipeline-step-result {
        background: var(--background-fill-primary);
        padding: 1.25rem;
        border-radius: 12px;
        border: 2px solid #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    .step-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--border-color-primary);
    }

    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background: #667eea;
        color: white;
        border-radius: 50%;
        font-weight: 700;
        font-size: 0.9rem;
        flex-shrink: 0;
        margin-top: 0.5rem;
        margin-left: 0.5rem;
    }

    .step-number.disabled {
        background: #adb5bd;
    }

    .step-title {
        font-weight: 700;
        color: var(--body-text-color);
        font-size: 1rem;
    }

    .step-content {
        min-height: 60px;
    }

    .pipeline-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #667eea;
        font-size: 1.8rem;
        font-weight: 700;
        padding: 0 0.5rem;
    }

    .input-controls-section {
        background: var(--background-fill-primary);
        padding: 1.25rem;
        border-radius: 12px;
        border: 2px solid var(--border-color-primary);
    }

    .section-label {
        font-weight: 700;
        font-size: 0.95rem;
        color: var(--body-text-color-subdued);
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .process-btn button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        border: none !important;
        font-size: 1rem !important;
        box-shadow: 0 2px 4px rgba(72, 187, 120, 0.3) !important;
        transition: all 0.2s !important;
        width: 100%;
    }

    .process-btn button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(72, 187, 120, 0.4) !important;
    }

    .preview-section {
        border: 3px solid #667eea;
        border-radius: 12px;
        padding: 1rem;
        background: var(--background-fill-primary);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
    }

    .preview-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: var(--body-text-color-subdued);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .class-output-inline {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        text-align: center !important;
        padding: 1.25rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        margin-bottom: 0.75rem !important;
    }

    .confidence-output-inline {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        text-align: center !important;
        padding: 1.25rem !important;
        background: var(--background-fill-secondary) !important;
        border-radius: 8px !important;
        color: var(--body-text-color) !important;
    }

    .info-section {
        background: var(--background-fill-primary);
        padding: 1.25rem;
        border-radius: 12px;
        border: 2px solid var(--border-color-primary);
        margin-top: 1rem;
    }

    .info-title {
        font-weight: 700;
        font-size: 1rem;
        color: var(--body-text-color);
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }

    .debug-log textarea {
        background: #1e1e1e !important;
        color: #d4d4d4 !important;
        font-family: 'Consolas', 'Monaco', monospace !important;
        font-size: 0.9rem !important;
        border-radius: 8px !important;
    }

    .gradio-radio {
        flex-direction: row !important;
        gap: 0.5rem !important;
    }

    .gradio-radio label {
        margin: 0 !important;
        padding: 0.4rem 1rem !important;
        border-radius: 6px !important;
        transition: all 0.2s !important;
    }

    .upload-container {
    height: 10px !important;
    font-size: 4px !important;
    }
    
    .upload-container > .image-frame {
    height: 0px !important;
    }

    .upload-container > button > .wrap {
    flex-direction: row !important;
    height: 10px !important;
    }
    """