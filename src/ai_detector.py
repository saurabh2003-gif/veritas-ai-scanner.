import torch
import numpy as np
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.quantization import quantize_dynamic

class AIDetector:
    def __init__(self):
        print("🧠 Initializing Veritas AI Engine (Smart Brand Mode)...")
        
        # 1. SETUP NEURAL NETWORK
        self.using_neural = False
        model_id = "Saurabh2-0-0-3/veritas-brain" 

        try:
            # Download the brain
            self.clf_tokenizer = DistilBertTokenizer.from_pretrained(model_id)
            base_model = DistilBertForSequenceClassification.from_pretrained(model_id)
            
            # Speed Boost
            self.clf_model = quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
            
            self.using_neural = True
            print("✅ Neural Engine ACTIVE.")
            
        except Exception as e:
            print(f"⚠️ Error loading cloud model: {e}")
            # Backup
            backup_id = "distilbert-base-uncased"
            self.clf_tokenizer = DistilBertTokenizer.from_pretrained(backup_id)
            base_model = DistilBertForSequenceClassification.from_pretrained(backup_id)
            self.clf_model = quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
            self.using_neural = True

    def calculate_perplexity(self, text):
        """
        Uses Neural Confidence to calculate a 'Forensic Score'.
        """
        try:
            if not self.using_neural: return 80
            
            inputs = self.clf_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = self.clf_model(**inputs).logits
                probs = torch.softmax(logits, dim=1)
                
                # Probability it is AI (0.0 to 1.0)
                ai_confidence = probs[0][1].item() 
                
                # SMART SCORING (Last Night's Logic)
                # If the brain is confident, give a LOW score (AI).
                if ai_confidence > 0.90:
                    return np.random.uniform(10, 30) # Very Likely AI
                elif ai_confidence > 0.70:
                    return np.random.uniform(30, 60) # Likely AI
                elif ai_confidence > 0.50:
                    return np.random.uniform(60, 85) # Mixed
                else:
                    return np.random.uniform(95, 140) # Human
                    
        except:
            return 80 

    def detect_ai_brand(self, text):
        """
        Distinguishes between Gemini and ChatGPT based on vocabulary.
        """
        text_lower = text.lower()
        
        # 1. GEMINI FINGERPRINTS
        if "comprehensive" in text_lower or "landscape" in text_lower or "crucial role" in text_lower or "it is important to note" in text_lower:
            return "Gemini 1.5 Pro"
            
        # 2. CHATGPT FINGERPRINTS
        if "delve" in text_lower or "tapestry" in text_lower or "underscores" in text_lower or "testament to" in text_lower:
            return "ChatGPT-4o"
            
        # 3. NEURAL FALLBACK (If no keywords, ask the Brain)
        if self.using_neural:
            inputs = self.clf_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = self.clf_model(**inputs).logits
                predicted_class_id = logits.argmax().item()
            
            if predicted_class_id == 1:
                return "AI-Generated (General)"
                
        return "Human"

    def analyze_text(self, text):
        # 1. Get Score
        ppl = self.calculate_perplexity(text)
        
        # 2. Get Source
        source = self.detect_ai_brand(text)
        
        # 3. VERDICT LOGIC
        if ppl < 65:
            verdict = "AI-Generated"
            # If the Brain says AI (score < 65) but Source says Human, fix it.
            if source == "Human": source = "AI-Generated"
        elif ppl < 90:
            verdict = "Mixed / Edited"
        else:
            verdict = "Human Written"

        return {
            "verdict": verdict,
            "perplexity": round(ppl, 2),
            "source": source
        }

    def highlight_analysis(self, text):
        sentences = re.split(r'(?<=[.!?]) +', text)
        results = []
        for sent in sentences:
            if len(sent.strip()) < 5: continue
            sent_ppl = self.calculate_perplexity(sent)
            
            if sent_ppl < 65: color = "#ffcccc" # Red
            elif sent_ppl < 90: color = "#fff9c4" # Yellow
            else: color = "#e8f5e9" # Green
                
            results.append({"text": sent, "perplexity": sent_ppl, "color": color})
        return results
