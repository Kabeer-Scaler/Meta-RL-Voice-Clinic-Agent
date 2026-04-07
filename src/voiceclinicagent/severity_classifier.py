"""
Severity classification system for VoiceClinicAgent.

This module provides a framework for the agent to learn disease severity
patterns rather than memorizing specific diseases.
"""

from typing import Dict, List, Any, Tuple
from enum import Enum


class SeverityLevel(Enum):
    """Severity levels for symptoms and conditions."""
    MINIMAL = 0.0  # 0.0-0.2: Minor issues, can wait days
    MILD = 0.25    # 0.2-0.4: Uncomfortable but not urgent
    MODERATE = 0.5  # 0.4-0.6: Needs attention soon
    SIGNIFICANT = 0.7  # 0.6-0.8: Should be seen today
    SEVERE = 0.85   # 0.8-0.9: Urgent, same-day required
    CRITICAL = 0.95  # 0.9-1.0: Emergency, immediate attention


class SymptomCategory(Enum):
    """Categories of symptoms for pattern learning."""
    RESPIRATORY = "respiratory"  # Breathing, cough, wheezing
    CARDIAC = "cardiac"  # Chest pain, palpitations, sweating
    NEUROLOGICAL = "neurological"  # Headache, dizziness, confusion
    GASTROINTESTINAL = "gastrointestinal"  # Nausea, vomiting, pain
    DERMATOLOGICAL = "dermatological"  # Rash, itching, swelling
    MUSCULOSKELETAL = "musculoskeletal"  # Joint pain, muscle ache
    INFECTIOUS = "infectious"  # Fever, chills, fatigue
    PEDIATRIC = "pediatric"  # Child-specific symptoms


class RiskFactor(Enum):
    """Risk factors that modify severity."""
    AGE_ELDERLY = "age_elderly"  # Age > 65
    AGE_INFANT = "age_infant"  # Age < 2
    CHRONIC_CARDIAC = "chronic_cardiac"  # Heart disease
    CHRONIC_RESPIRATORY = "chronic_respiratory"  # Asthma, COPD
    CHRONIC_METABOLIC = "chronic_metabolic"  # Diabetes
    IMMUNOCOMPROMISED = "immunocompromised"  # Weak immune system
    PREGNANCY = "pregnancy"
    PRIOR_ESCALATION = "prior_escalation"  # History of urgent visits


class SeverityClassifier:
    """
    Classifies symptom severity based on multiple factors.
    
    This system teaches the agent to learn PATTERNS:
    - Symptom combinations
    - Risk factor interactions
    - Temporal progression
    - Context-dependent severity
    
    NOT specific diseases!
    """
    
    # Symptom keywords mapped to categories and base severity
    SYMPTOM_PATTERNS = {
        # Respiratory symptoms
        "difficulty breathing": (SymptomCategory.RESPIRATORY, 0.8),
        "shortness of breath": (SymptomCategory.RESPIRATORY, 0.75),
        "wheezing": (SymptomCategory.RESPIRATORY, 0.6),
        "cough": (SymptomCategory.RESPIRATORY, 0.3),
        "chest tightness": (SymptomCategory.RESPIRATORY, 0.65),
        
        # Cardiac symptoms
        "chest pain": (SymptomCategory.CARDIAC, 0.85),
        "chest pressure": (SymptomCategory.CARDIAC, 0.8),
        "palpitations": (SymptomCategory.CARDIAC, 0.6),
        "sweating": (SymptomCategory.CARDIAC, 0.4),
        "dizziness": (SymptomCategory.CARDIAC, 0.5),
        
        # Neurological symptoms
        "confusion": (SymptomCategory.NEUROLOGICAL, 0.85),
        "severe headache": (SymptomCategory.NEUROLOGICAL, 0.7),
        "headache": (SymptomCategory.NEUROLOGICAL, 0.3),
        "numbness": (SymptomCategory.NEUROLOGICAL, 0.75),
        "vision changes": (SymptomCategory.NEUROLOGICAL, 0.7),
        
        # Gastrointestinal symptoms
        "severe abdominal pain": (SymptomCategory.GASTROINTESTINAL, 0.75),
        "vomiting": (SymptomCategory.GASTROINTESTINAL, 0.5),
        "nausea": (SymptomCategory.GASTROINTESTINAL, 0.3),
        "diarrhea": (SymptomCategory.GASTROINTESTINAL, 0.4),
        
        # Dermatological symptoms
        "rash": (SymptomCategory.DERMATOLOGICAL, 0.2),
        "severe rash": (SymptomCategory.DERMATOLOGICAL, 0.5),
        "swelling": (SymptomCategory.DERMATOLOGICAL, 0.6),
        "itching": (SymptomCategory.DERMATOLOGICAL, 0.15),
        
        # Infectious symptoms
        "high fever": (SymptomCategory.INFECTIOUS, 0.6),
        "fever": (SymptomCategory.INFECTIOUS, 0.4),
        "chills": (SymptomCategory.INFECTIOUS, 0.3),
    }
    
    # Risk factors and their severity multipliers
    RISK_MULTIPLIERS = {
        RiskFactor.AGE_ELDERLY: 1.3,
        RiskFactor.AGE_INFANT: 1.4,
        RiskFactor.CHRONIC_CARDIAC: 1.5,
        RiskFactor.CHRONIC_RESPIRATORY: 1.3,
        RiskFactor.CHRONIC_METABOLIC: 1.2,
        RiskFactor.IMMUNOCOMPROMISED: 1.4,
        RiskFactor.PREGNANCY: 1.3,
        RiskFactor.PRIOR_ESCALATION: 1.2,
    }
    
    # Symptom combinations that increase severity
    DANGEROUS_COMBINATIONS = [
        # Cardiac + respiratory = very serious
        ([SymptomCategory.CARDIAC, SymptomCategory.RESPIRATORY], 1.5),
        # Neurological + cardiac = stroke risk
        ([SymptomCategory.NEUROLOGICAL, SymptomCategory.CARDIAC], 1.6),
        # Respiratory + infectious in children = serious
        ([SymptomCategory.RESPIRATORY, SymptomCategory.INFECTIOUS], 1.3),
    ]
    
    @classmethod
    def classify_severity(
        cls,
        symptoms: List[str],
        chronic_conditions: List[str],
        age: int,
        prior_escalations: List[Dict],
        duration_days: int,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Classify severity based on multiple factors.
        
        This forces the agent to learn:
        1. Symptom pattern recognition
        2. Risk factor assessment
        3. Combination effects
        4. Temporal dynamics
        
        Args:
            symptoms: List of symptom descriptions
            chronic_conditions: List of chronic conditions
            age: Patient age
            prior_escalations: Prior urgent visits
            duration_days: How long symptoms have lasted
            
        Returns:
            Tuple of (severity_score, explanation_dict)
        """
        # Extract symptom categories and base severities
        detected_symptoms = []
        base_severity = 0.0
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            for pattern, (category, severity) in cls.SYMPTOM_PATTERNS.items():
                if pattern in symptom_lower:
                    detected_symptoms.append((category, severity))
                    base_severity = max(base_severity, severity)
        
        if not detected_symptoms:
            # No recognized symptoms, use default
            base_severity = 0.3
        
        # Identify risk factors
        risk_factors = cls._identify_risk_factors(
            chronic_conditions, age, prior_escalations
        )
        
        # Apply risk multipliers
        severity_with_risk = base_severity
        for risk_factor in risk_factors:
            multiplier = cls.RISK_MULTIPLIERS.get(risk_factor, 1.0)
            severity_with_risk *= multiplier
        
        # Check for dangerous combinations
        categories = [cat for cat, _ in detected_symptoms]
        for combo, multiplier in cls.DANGEROUS_COMBINATIONS:
            if all(cat in categories for cat in combo):
                severity_with_risk *= multiplier
        
        # Temporal factor: symptoms getting worse over time
        if duration_days > 7:
            severity_with_risk *= 1.1  # Persistent symptoms
        elif duration_days < 1:
            severity_with_risk *= 1.2  # Acute onset
        
        # Clamp to [0.0, 1.0]
        final_severity = min(1.0, severity_with_risk)
        
        # Build explanation for interpretability
        explanation = {
            "base_severity": base_severity,
            "detected_symptoms": [(cat.value, sev) for cat, sev in detected_symptoms],
            "risk_factors": [rf.value for rf in risk_factors],
            "final_severity": final_severity,
            "severity_level": cls._get_severity_level(final_severity),
        }
        
        return final_severity, explanation
    
    @classmethod
    def _identify_risk_factors(
        cls,
        chronic_conditions: List[str],
        age: int,
        prior_escalations: List[Dict],
    ) -> List[RiskFactor]:
        """Identify risk factors from patient data."""
        risk_factors = []
        
        # Age-based risk
        if age >= 65:
            risk_factors.append(RiskFactor.AGE_ELDERLY)
        elif age <= 2:
            risk_factors.append(RiskFactor.AGE_INFANT)
        
        # Chronic condition risk
        for condition in chronic_conditions:
            condition_lower = condition.lower()
            if any(term in condition_lower for term in ["heart", "cardiac", "hypertension"]):
                risk_factors.append(RiskFactor.CHRONIC_CARDIAC)
            if any(term in condition_lower for term in ["asthma", "copd", "respiratory"]):
                risk_factors.append(RiskFactor.CHRONIC_RESPIRATORY)
            if any(term in condition_lower for term in ["diabetes", "metabolic"]):
                risk_factors.append(RiskFactor.CHRONIC_METABOLIC)
        
        # Prior escalation risk
        if prior_escalations:
            risk_factors.append(RiskFactor.PRIOR_ESCALATION)
        
        return risk_factors
    
    @classmethod
    def _get_severity_level(cls, severity_score: float) -> str:
        """Map severity score to level name."""
        if severity_score >= 0.9:
            return "CRITICAL"
        elif severity_score >= 0.8:
            return "SEVERE"
        elif severity_score >= 0.6:
            return "SIGNIFICANT"
        elif severity_score >= 0.4:
            return "MODERATE"
        elif severity_score >= 0.2:
            return "MILD"
        else:
            return "MINIMAL"
    
    @classmethod
    def compute_deterioration_adjusted_severity(
        cls,
        base_severity: float,
        deterioration_state: Dict[str, Any],
    ) -> float:
        """
        Adjust severity based on deterioration dynamics.
        
        This teaches the agent temporal reasoning:
        - Symptoms worsening = higher severity
        - Stable symptoms = maintain severity
        - Improving symptoms = lower severity
        
        Args:
            base_severity: Initial severity score
            deterioration_state: Current deterioration state
            
        Returns:
            Adjusted severity score
        """
        if not deterioration_state.get("enabled", False):
            return base_severity
        
        current_urgency = deterioration_state.get("current_urgency", base_severity)
        stage = deterioration_state.get("stage", "stable")
        
        # Stage-based adjustments
        stage_multipliers = {
            "stable": 1.0,
            "mild_worsening": 1.2,
            "moderate_worsening": 1.4,
            "critical": 1.6,
        }
        
        multiplier = stage_multipliers.get(stage, 1.0)
        adjusted = min(1.0, current_urgency * multiplier)
        
        return adjusted


def generate_severity_features(observation: Dict[str, Any]) -> Dict[str, float]:
    """
    Generate severity-related features for the agent to learn from.
    
    This creates a feature vector that the agent's neural network can use
    to learn severity patterns across different diseases.
    
    Args:
        observation: Current observation dict
        
    Returns:
        Dict of severity features
    """
    features = {}
    
    # Extract patient message and clinical history
    patient_message = observation.get("patient_message", "")
    clinical_history = observation.get("clinical_history", {})
    patient_flags = observation.get("patient_flags", {})
    
    # Symptom category presence (binary features)
    for category in SymptomCategory:
        features[f"has_{category.value}_symptoms"] = 0.0
    
    # Check for symptom patterns
    message_lower = patient_message.lower()
    for pattern, (category, _) in SeverityClassifier.SYMPTOM_PATTERNS.items():
        if pattern in message_lower:
            features[f"has_{category.value}_symptoms"] = 1.0
    
    # Risk factor presence (binary features)
    chronic_conditions = clinical_history.get("chronic_conditions", [])
    for condition in chronic_conditions:
        condition_lower = condition.lower()
        if "heart" in condition_lower or "cardiac" in condition_lower:
            features["has_cardiac_risk"] = 1.0
        if "asthma" in condition_lower or "respiratory" in condition_lower:
            features["has_respiratory_risk"] = 1.0
        if "diabetes" in condition_lower:
            features["has_metabolic_risk"] = 1.0
    
    # High risk flag
    features["high_risk_flag"] = 1.0 if clinical_history.get("high_risk_flag", False) else 0.0
    
    # Prior escalations
    prior_escalations = clinical_history.get("prior_escalations", [])
    features["has_prior_escalations"] = 1.0 if prior_escalations else 0.0
    
    # Urgency hint (continuous feature)
    features["urgency_hint"] = patient_flags.get("urgency_score_hint", 0.5)
    
    # Symptom combination features (interaction terms)
    has_cardiac = features.get("has_cardiac_symptoms", 0.0)
    has_respiratory = features.get("has_respiratory_symptoms", 0.0)
    features["cardiac_respiratory_combo"] = has_cardiac * has_respiratory
    
    return features
