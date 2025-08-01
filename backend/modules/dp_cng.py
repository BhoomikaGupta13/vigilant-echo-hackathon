# dp_cng.py
# Dynamic Persona-Based Counter-Narrative Generator (Highly Simplified for 6-Hour Hackathon)

async def generate_counter_narrative(analysis_results: dict):
    """
    Generates a simple counter-narrative based on the analysis results.
    For Round 1, this is rule-based, not an LLM.
    """
    cm_sdd_results = analysis_results.get("cm_sdd", {})
    ls_zlf_results = analysis_results.get("ls_zlf", {})

    # Determine if any major issues were detected
    is_deepfake_conceptual = ls_zlf_results.get("deepfake_analysis", {}).get("deepfake_detected", False)
    is_semantic_discrepant = cm_sdd_results.get("discrepancy_detected", False)

    llm_origin = ls_zlf_results.get("llm_origin_analysis", {}).get("llm_origin", "N/A")
    is_llm_generated_detected = llm_origin not in ["N/A", "Human/Uncertain"]

    reasons = []
    if is_semantic_discrepant:
        reasons.append("Cross-modal inconsistencies detected.")
        reasons.extend(cm_sdd_results.get("discrepancy_reason", []))
    if is_deepfake_conceptual: # This will likely be False for Round 1
        reasons.append("Potential deepfake content identified.")
    if is_llm_generated_detected:
        reasons.append(f"Content appears to be AI-generated by {llm_origin}.")

    # Generate a general counter-narrative based on severity
    if is_semantic_discrepant or is_deepfake_conceptual:
        return f"🚨 ALERT: This content has critical inconsistencies ({', '.join(reasons[:2])}). Please verify information with trusted sources."
    elif is_llm_generated_detected:
        return f"💡 This content is likely AI-generated by {llm_origin}. Consider its origin and context."
    else:
        return "✅ Content appears consistent. No major misinformation or deepfake indicators found."